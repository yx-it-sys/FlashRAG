#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from statistics import mean

from common import (
    DenseSimilarity,
    UTILITY_TO_SCORE,
    classify_facts_by_novelty,
    normalize_whitespace,
    read_jsonl,
    resolve_retriever_config_path,
    write_json,
    write_jsonl_incremental,
)

FACTS_INPUT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/scripts/rte_robustness/split/qwen2_5_7b/fact_outputs.jsonl"
)
UTILITY_INPUT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/scripts/rte_robustness/split/qwen2_5_7b/utility_outputs.jsonl"
)
OUTPUT_JSONL = Path(
    "/home/you/FlashRAG/exps/idea10/scripts/rte_robustness/qwen2_5_qwen2_5/rte_outputs.jsonl"
)
OUTPUT_SUMMARY = Path(
    "/home/you/FlashRAG/exps/idea10/scripts/rte_robustness/qwen2_5_qwen2_5/rte_outputs.summary.json"
)
HUMAN_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/cross-model-rte/label/human/TQS/human_label_150.jsonl"
)
SPEARMAN_OUTPUT_PATH = OUTPUT_JSONL.with_name("spearman.txt")

SIM_THRESHOLD = 1.0
FACT_SIM_THRESHOLD = 0.9
EPSILON = 1e-8
EMBED_BATCH_SIZE = 64
HUMAN_QUALITY_TO_SCORE = {
    "Low: mostly unhelpful trajectory": 1.0,
    "Medium: mixed progress": 2.0,
    "High: consistently useful trajectory": 3.0,
}


def build_utility_lookup(items: list[dict]) -> dict[tuple[str, int], dict]:
    lookup = {}
    for sample in items:
        sample_id = sample.get("id")
        for step in sample.get("steps", []):
            lookup[(sample_id, int(step["iteration_index"]))] = step
    return lookup


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearsonr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        raise ValueError("Spearman correlation is undefined when one variable has zero variance.")
    return cov / math.sqrt(var_x * var_y)


def write_spearman_report(
    human_path: Path,
    model_path: Path,
    output_path: Path,
) -> None:
    human_items = read_jsonl(human_path)
    model_items = read_jsonl(model_path)

    human_scores_by_id = {}
    for item in human_items:
        sample_id = item["sample_id"]
        human_label = item["trajectory_labels"]["trajectory_quality"]
        if human_label not in HUMAN_QUALITY_TO_SCORE:
            raise KeyError(f"Unsupported human trajectory_quality label: {human_label!r}")
        human_scores_by_id[sample_id] = HUMAN_QUALITY_TO_SCORE[human_label]

    paired_ids = []
    human_scores = []
    model_scores = []
    for item in model_items:
        sample_id = item["id"]
        if sample_id not in human_scores_by_id:
            continue
        paired_ids.append(sample_id)
        human_scores.append(human_scores_by_id[sample_id])
        model_scores.append(float(item["trajectory_quality_score"]))

    if not paired_ids:
        raise ValueError("No overlapping sample ids found between human and model files.")

    spearman = pearsonr(rankdata(human_scores), rankdata(model_scores))
    missing_human_ids = sorted(set(human_scores_by_id) - set(paired_ids))
    missing_model_ids = sorted({item["id"] for item in model_items} - set(paired_ids))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                f"human_path: {human_path}",
                f"model_path: {model_path}",
                "human_mapping: Low=1, Medium=2, High=3",
                f"num_pairs: {len(paired_ids)}",
                f"spearman_rho: {spearman:.12f}",
                f"missing_human_ids: {len(missing_human_ids)}",
                f"missing_model_ids: {len(missing_model_ids)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {output_path}")
    print(f"spearman_rho={spearman:.12f}")


def compute_rte_outputs() -> None:
    facts_items = read_jsonl(FACTS_INPUT_PATH)
    utility_items = read_jsonl(UTILITY_INPUT_PATH)
    utility_lookup = build_utility_lookup(utility_items)
    dense_similarity = DenseSimilarity(
        config_path=resolve_retriever_config_path(type("Args", (), {})()),
        batch_size=EMBED_BATCH_SIZE,
    )

    results = []
    for sample in facts_items:
        sample_id = sample.get("id")
        prev_query = None
        cumulative_facts: list[str] = []
        step_rows = []
        for step in sample.get("steps", []):
            query = step.get("query") or sample.get("question", "")
            sub_question = normalize_whitespace(step.get("sub_question", "")) or normalize_whitespace(query)
            reaction_text = normalize_whitespace(step.get("reaction_text", ""))
            if not reaction_text:
                continue
            use_query_similarity = step["retrieval_action"] != "image_retrieval_result"
            query_sim = (
                dense_similarity.similarity(prev_query, query)
                if use_query_similarity and prev_query and query
                else None
            )
            if use_query_similarity and query_sim is not None and query_sim > SIM_THRESHOLD:
                facts = []
                accepted_facts = []
                novel_facts = []
                fact_match_rows = []
                delta_f = 0.0
                delta_reason = "similarity_gate"
                utility_label = "no_contribution"
                utility_reason = "Similarity gate triggered because query similarity exceeded the threshold."
                utility_defined = False
                utility_reason_code = "similarity_gate"
                utility = 0.0
                filtered_facts = []
            else:
                facts = list(step.get("facts_t", []))
                accepted_facts, novel_facts, fact_match_rows = classify_facts_by_novelty(
                    facts,
                    cumulative_facts,
                    dense_similarity,
                    FACT_SIM_THRESHOLD,
                )
                delta_f = len(novel_facts) / (len(facts) + EPSILON)
                delta_reason = "set_difference" if use_query_similarity else "set_difference_no_query_similarity"
                utility_row = utility_lookup.get((sample_id, int(step["iteration_index"])))
                if utility_row is None:
                    if use_query_similarity:
                        prev_query = query
                    print(f"[RTE] skip id={sample_id} iter={step['iteration_index']} missing utility", flush=True)
                    continue
                utility_label = utility_row["utility_label"]
                utility_reason = utility_row["utility_reason"]
                utility_defined = True
                utility_reason_code = "llm_three_level_utility"
                utility = float(utility_row["utility"])
                filtered_facts = []
                cumulative_facts.extend(accepted_facts)
            step_score = utility * delta_f
            step_rows.append(
                {
                    "iteration_index": step["iteration_index"],
                    "retrieval_action": step["retrieval_action"],
                    "retrieval_type": step["retrieval_type"],
                    "mode": step["mode"],
                    "previous_query_for_similarity": prev_query,
                    "query": step["query"],
                    "sub_question": sub_question,
                    "query_similarity_enabled": use_query_similarity,
                    "query_similarity_with_previous": query_sim,
                    "similarity_threshold": SIM_THRESHOLD,
                    "fact_similarity_threshold": FACT_SIM_THRESHOLD,
                    "retrieval_index": step["retrieval_index"],
                    "evidence_source": step["evidence_source"],
                    "evidence_text": step["evidence_text"],
                    "reaction_index": step["reaction_index"],
                    "reaction_action": step["reaction_action"],
                    "reaction_text": reaction_text,
                    "facts_t": facts,
                    "filtered_facts_t": filtered_facts,
                    "accepted_facts_t": accepted_facts,
                    "delta_f": delta_f,
                    "novel_facts": novel_facts,
                    "fact_novelty_matches": fact_match_rows,
                    "cumulative_fact_count_after_dedup": len(cumulative_facts),
                    "delta_reason": delta_reason,
                    "utility_label": utility_label,
                    "utility_defined": utility_defined,
                    "utility_reason_code": utility_reason_code,
                    "utility": utility,
                    "utility_reason": utility_reason,
                    "step_score": step_score,
                }
            )
            if use_query_similarity:
                prev_query = query
        results.append(
            {
                "id": sample_id,
                "question": sample.get("question"),
                "status": sample.get("status"),
                "num_iterations": len(step_rows),
                "total_delta_f": sum(item["delta_f"] for item in step_rows),
                "total_step_score": sum(item["step_score"] for item in step_rows),
                "trajectory_quality_score": mean(item["step_score"] for item in step_rows) if step_rows else 0.0,
                "steps": step_rows,
            }
        )
        print(f"[RTE] completed id={sample_id} iters={len(step_rows)}", flush=True)

    steps = [step for sample in results for step in sample["steps"]]
    write_jsonl_incremental(OUTPUT_JSONL, results)
    write_json(
        OUTPUT_SUMMARY,
        {
            "meta": {
                "facts_input_path": str(FACTS_INPUT_PATH),
                "utility_input_path": str(UTILITY_INPUT_PATH),
                "output_jsonl": str(OUTPUT_JSONL),
                "similarity_threshold": SIM_THRESHOLD,
                "fact_similarity_threshold": FACT_SIM_THRESHOLD,
                "epsilon": EPSILON,
                "num_samples": len(results),
                "num_steps": len(steps),
                "utility_mapping": UTILITY_TO_SCORE,
            },
            "aggregate": {
                "mean_tqs": mean(item["trajectory_quality_score"] for item in results) if results else 0.0,
                "mean_iterations": mean(item["num_iterations"] for item in results) if results else 0.0,
                "mean_delta_f": mean(step["delta_f"] for step in steps) if steps else 0.0,
                "mean_utility": mean(step["utility"] for step in steps) if steps else 0.0,
                "mean_step_score": mean(step["step_score"] for step in steps) if steps else 0.0,
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rte", action="store_true")
    parser.add_argument("--skip-spearman", action="store_true")
    parser.add_argument("--human-path", type=Path, default=HUMAN_PATH)
    parser.add_argument("--spearman-model-path", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--spearman-output-path", type=Path, default=SPEARMAN_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_rte:
        compute_rte_outputs()
    if not args.skip_spearman:
        write_spearman_report(
            human_path=args.human_path,
            model_path=args.spearman_model_path,
            output_path=args.spearman_output_path,
        )


if __name__ == "__main__":
    main()
