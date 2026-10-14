#!/usr/bin/env python3
import os
from pathlib import Path
from statistics import mean

from evaluate_trajectory_quality_for_test import (
    append_jsonl,
    DEFAULT_FACT_PROMPT,
    DEFAULT_FACT_SIM_THRESHOLD,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    DenseSimilarity,
    LLMJudge,
    build_iterations,
    build_local_fact_generator,
    build_sample_id,
    classify_facts_by_novelty,
    load_prompt,
    normalize_whitespace,
    read_jsonl,
    resolve_retriever_config_path,
    resolve_run_config_path,
    write_json,
    write_jsonl,
)


DEFAULT_INPUT = Path(
    os.environ.get(
        "TRAJECTORY_FACT_INPUT",
        "/home/you/FlashRAG/exps/idea10/data/result/cross-model-rte/label/human/TQS/human_label_150.trajectories.jsonl",
    )
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "TRAJECTORY_FACT_OUTPUT_DIR",
        "/home/you/FlashRAG/exps/idea10/data/result/cross-model-rte/label/human/TQS/llama_fact_stage",
    )
)
DEFAULT_LOCAL_FACT_MODEL = os.environ.get(
    "TRAJECTORY_FACT_MODEL_NAME",
    "Meta-Llama-3-8B-Instruct",
)
DEFAULT_LOCAL_FACT_MODEL_PATH = Path(
    os.environ.get(
        "TRAJECTORY_FACT_MODEL_PATH",
        "/mnt/data/you/modelscope/Meta-Llama-3-8B-Instruct",
    )
)
DEFAULT_SIM_THRESHOLD = 1.0
DEFAULT_EPSILON = 1e-8
DEFAULT_MAX_DOCS_PER_ITERATION = 10
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_CACHE_FLUSH_EVERY = 1


class Args:
    input = DEFAULT_INPUT
    output_dir = DEFAULT_OUTPUT_DIR
    fact_prompt = DEFAULT_FACT_PROMPT
    fact_extractor = "local_qwen"
    local_fact_model = DEFAULT_LOCAL_FACT_MODEL
    local_fact_model_path = DEFAULT_LOCAL_FACT_MODEL_PATH
    fact_api_model = "Qwen/Qwen2.5-7B-Instruct"
    model = "Qwen/Qwen2.5-7B-Instruct"
    timeout = DEFAULT_TIMEOUT
    retries = DEFAULT_RETRIES
    fact_sim_threshold = DEFAULT_FACT_SIM_THRESHOLD
    sim_threshold = DEFAULT_SIM_THRESHOLD
    epsilon = DEFAULT_EPSILON
    max_docs_per_iteration = DEFAULT_MAX_DOCS_PER_ITERATION
    embed_batch_size = DEFAULT_EMBED_BATCH_SIZE
    cache_flush_every = DEFAULT_CACHE_FLUSH_EVERY
    utility_prompt = None
    utility_judge_backend = "api"
    local_utility_model = "Qwen2.5-7B-Instruct"
    local_utility_model_path = Path("/mnt/data/you/modelscope/Qwen2.5-7B-Instruct")
    api_key_env = "SILICONFLOW_API_KEY"
    base_url = "https://api.siliconflow.cn/v1"
    log = None


def extract_fact_sample(
    sample: dict,
    judge: LLMJudge,
    sim_threshold: float,
    fact_sim_threshold: float,
    epsilon: float,
    dense_similarity: DenseSimilarity,
) -> dict:
    sample_id = build_sample_id(sample)
    iterations = build_iterations(sample.get("trajectory", []))
    prev_query = None
    cumulative_facts: list[str] = []
    step_rows = []

    for iteration in iterations:
        if iteration["retrieval_action"] == "no_retrieval_result":
            continue
        query = iteration["query"] or sample.get("question", "")
        sub_question = normalize_whitespace(iteration.get("sub_question", ""))
        reaction_text = normalize_whitespace(iteration.get("reaction_text", ""))
        evidence_text = normalize_whitespace(iteration.get("retrieval_content", ""))
        evidence_source = iteration.get("evidence_source", "explicit_retrieval_result")
        if not evidence_text and reaction_text:
            evidence_text = reaction_text
            evidence_source = "reaction_text_fallback"
        if not sub_question:
            sub_question = normalize_whitespace(query)
        if not reaction_text:
            continue

        use_query_similarity = iteration["retrieval_action"] != "image_retrieval_result"
        retrieval_type = (
            "Image Retrieval"
            if iteration["retrieval_action"] == "image_retrieval_result"
            else "Text Retrieval"
        )
        query_sim = (
            dense_similarity.similarity(prev_query, query)
            if use_query_similarity and prev_query and query
            else None
        )

        if use_query_similarity and query_sim is not None and query_sim > sim_threshold:
            facts = []
            accepted_facts = []
            fact_match_rows = []
            delta_f = 0.0
            novel_facts = []
            delta_reason = "similarity_gate"
        else:
            facts = judge.extract_facts(
                query=query,
                evidence=evidence_text,
                retrieval_type=retrieval_type,
            )
            accepted_facts, novel_facts, fact_match_rows = classify_facts_by_novelty(
                facts,
                cumulative_facts,
                dense_similarity,
                fact_sim_threshold,
            )
            delta_f = len(novel_facts) / (len(facts) + epsilon)
            delta_reason = (
                "set_difference" if use_query_similarity else "set_difference_no_query_similarity"
            )
            cumulative_facts.extend(accepted_facts)

        step_rows.append(
            {
                "iteration_index": iteration["iteration_index"],
                "retrieval_action": iteration["retrieval_action"],
                "retrieval_type": retrieval_type,
                "mode": iteration["mode"],
                "previous_query_for_similarity": prev_query,
                "query": iteration["query"],
                "sub_question": sub_question,
                "query_similarity_enabled": use_query_similarity,
                "query_similarity_with_previous": query_sim,
                "similarity_threshold": sim_threshold,
                "fact_similarity_threshold": fact_sim_threshold,
                "retrieval_index": iteration["retrieval_index"],
                "evidence_source": evidence_source,
                "evidence_text": evidence_text,
                "reaction_index": iteration["reaction_index"],
                "reaction_action": iteration["reaction_action"],
                "reaction_text": reaction_text,
                "facts_t": facts,
                "accepted_facts_t": accepted_facts,
                "delta_f": delta_f,
                "novel_facts": novel_facts,
                "fact_novelty_matches": fact_match_rows,
                "cumulative_fact_count_after_dedup": len(cumulative_facts),
                "delta_reason": delta_reason,
            }
        )
        if use_query_similarity:
            prev_query = query

    return {
        "id": sample_id,
        "question": sample.get("question"),
        "status": sample.get("status"),
        "num_iterations": len(step_rows),
        "total_delta_f": sum(step["delta_f"] for step in step_rows),
        "steps": step_rows,
    }


def summarize(results: list[dict]) -> dict:
    steps = [step for sample in results for step in sample["steps"]]
    return {
        "meta": {
            "input": str(DEFAULT_INPUT),
            "output_dir": str(DEFAULT_OUTPUT_DIR),
            "local_fact_model": DEFAULT_LOCAL_FACT_MODEL,
            "local_fact_model_path": str(DEFAULT_LOCAL_FACT_MODEL_PATH),
            "similarity_threshold": DEFAULT_SIM_THRESHOLD,
            "fact_similarity_threshold": DEFAULT_FACT_SIM_THRESHOLD,
            "epsilon": DEFAULT_EPSILON,
            "num_samples": len(results),
            "num_steps": len(steps),
        },
        "aggregate": {
            "mean_iterations": mean(item["num_iterations"] for item in results) if results else 0.0,
            "mean_delta_f": mean(step["delta_f"] for step in steps) if steps else 0.0,
            "informative_step_rate": (sum(1 for step in steps if step["delta_f"] > 0) / len(steps)) if steps else 0.0,
        },
    }


def main() -> None:
    args = Args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[Fact Stage] input={args.input} output_dir={args.output_dir} "
        f"fact_model={args.local_fact_model} fact_model_path={args.local_fact_model_path}",
        flush=True,
    )
    fact_prompt = load_prompt(args.fact_prompt)
    dense_similarity = DenseSimilarity(
        config_path=resolve_retriever_config_path(args),
        batch_size=args.embed_batch_size,
    )
    local_fact_generator = build_local_fact_generator(args, resolve_run_config_path(args))
    judge = LLMJudge(
        client=None,
        model=args.model,
        fact_api_model=args.fact_api_model,
        timeout=args.timeout,
        retries=args.retries,
        max_docs_per_iteration=args.max_docs_per_iteration,
        fact_extractor=args.fact_extractor,
        local_fact_generator=local_fact_generator,
        utility_judge_backend=args.utility_judge_backend,
        local_utility_generator=None,
        fact_prompt=fact_prompt,
        utility_prompt="",
        fact_cache_path=args.output_dir / "fact_cache.json",
        utility_cache_path=args.output_dir / "utility_cache_unused.json",
        cache_flush_every=args.cache_flush_every,
    )

    samples = read_jsonl(args.input)
    output_path = args.output_dir / "trajectory_facts.jsonl"
    summary_path = args.output_dir / "trajectory_facts_summary.json"
    if output_path.exists():
        output_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    results = []
    total = len(samples)
    for idx, sample in enumerate(samples, start=1):
        result = extract_fact_sample(
            sample=sample,
            judge=judge,
            sim_threshold=args.sim_threshold,
            fact_sim_threshold=args.fact_sim_threshold,
            epsilon=args.epsilon,
            dense_similarity=dense_similarity,
        )
        results.append(result)
        append_jsonl(output_path, result)
        print(
            f"[Fact Stage] completed {idx}/{total} id={result['id']} "
            f"iters={result['num_iterations']} total_delta_f={result['total_delta_f']:.4f}",
            flush=True,
        )
    judge.flush_caches()
    write_json(summary_path, summarize(results))


if __name__ == "__main__":
    main()
