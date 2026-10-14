#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

VALID_LABELS = {
    "Object Identification",
    "Description",
    "Indirect Entity Ambiguity",
    "No Object Involved",
}
LABEL_ALIASES = {
    "Normal ambiguouty": "Indirect Entity Ambiguity",
}

DEFAULT_BASE = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label"
)
DEFAULT_MANUAL_SAMPLE = DEFAULT_BASE / "human/trajectory_annotation_manual_sample.jsonl"
DEFAULT_HUMAN = DEFAULT_BASE / "human/annotation_results_readable.jsonl"
DEFAULT_LLM = DEFAULT_BASE / "qwen/trajectory_annotation.llm_labeled.adjudicated.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Cohen's Kappa between human and LLM entity-ambiguity labels "
            "for the overlapping query steps."
        )
    )
    parser.add_argument("--manual-sample", type=Path, default=DEFAULT_MANUAL_SAMPLE)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--llm", type=Path, default=DEFAULT_LLM)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON summary instead of text output.",
    )
    parser.add_argument(
        "--show-confusion",
        action="store_true",
        help="Include confusion matrices in text output.",
    )
    parser.add_argument(
        "--show-excluded",
        action="store_true",
        help="Include excluded / unlabeled overlap examples in text output.",
    )
    parser.add_argument(
        "--show-disagreements",
        action="store_true",
        help="Include disagreement examples in text output.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def canonicalize_label(entity: str | None, level: str | None) -> tuple[str, str | None] | None:
    level = LABEL_ALIASES.get(level, level)
    if entity == "No":
        return ("No", None)
    if entity == "Yes" and level in VALID_LABELS:
        return ("Yes", level)
    return None


def build_manual_step_sequence(manual_sample_items: list[dict]) -> dict[str, list[int]]:
    sequence_by_sample = {}
    for item in manual_sample_items:
        sample_id = item["annotation_id"]
        sequence_by_sample[sample_id] = [step["step_index"] for step in item.get("query_steps", [])]
    return sequence_by_sample


def build_human_map(human_items: list[dict], step_sequence: dict[str, list[int]]) -> dict[tuple[str, int], tuple[str, str | None] | None]:
    human_map = {}
    for item in human_items:
        sample_id = item["sample_id"]
        if sample_id not in step_sequence:
            continue
        steps = step_sequence[sample_id]
        annotations = item.get("query_annotations", [])
        if len(annotations) != len(steps):
            raise ValueError(
                f"Length mismatch for {sample_id}: human has {len(annotations)} queries, "
                f"manual sample has {len(steps)}"
            )
        for idx, annotation in enumerate(annotations):
            original_step_index = steps[idx]
            human_map[(sample_id, original_step_index)] = canonicalize_label(
                annotation.get("entity_ambiguous"),
                annotation.get("ambiguity_level"),
            )
    return human_map


def build_llm_map(llm_items: list[dict]) -> tuple[
    dict[tuple[str, int], tuple[str, str | None] | None],
    dict[tuple[str, int], dict],
]:
    llm_map = {}
    metadata = {}
    for item in llm_items:
        sample_id = item["annotation_id"]
        for step in item.get("query_steps", []):
            key = (sample_id, step["step_index"])
            llm_label = step.get("llm_label")
            llm_map[key] = canonicalize_label(
                llm_label.get("entity_ambiguous") if isinstance(llm_label, dict) else None,
                llm_label.get("ambiguity_level") if isinstance(llm_label, dict) else None,
            )
            metadata[key] = {
                "text_query": step.get("text_query"),
                "sub_question": step.get("sub_question"),
                "raw_llm_label": llm_label,
            }
    return llm_map, metadata


def compute_kappa(
    pairs: list[tuple[str, str]],
    labels: list[str],
) -> tuple[float, float, float, Counter, Counter, Counter]:
    total = len(pairs)
    if total == 0:
        raise ValueError("No valid aligned label pairs found.")
    observed = sum(1 for left, right in pairs if left == right) / total
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum((left_counts[label] / total) * (right_counts[label] / total) for label in labels)
    kappa = (observed - expected) / (1 - expected)
    confusion = Counter(pairs)
    return kappa, observed, expected, left_counts, right_counts, confusion


def summarize(manual_sample: list[dict], human_map: dict, llm_map: dict, llm_meta: dict) -> dict:
    common_keys = sorted(set(human_map) & set(llm_map))
    binary_pairs = []
    fine_pairs = []
    invalid_pairs = []
    disagreement_examples = []

    for key in common_keys:
        human_label = human_map[key]
        llm_label = llm_map[key]
        if human_label is None or llm_label is None:
            invalid_pairs.append(
                {
                    "annotation_id": key[0],
                    "step_index": key[1],
                    "human": human_label,
                    "llm_raw": llm_meta[key]["raw_llm_label"],
                    "text_query": llm_meta[key]["text_query"],
                    "sub_question": llm_meta[key]["sub_question"],
                }
            )
            continue

        human_binary = human_label[0]
        llm_binary = llm_label[0]
        human_fine = "No" if human_label[0] == "No" else human_label[1]
        llm_fine = "No" if llm_label[0] == "No" else llm_label[1]

        binary_pairs.append((human_binary, llm_binary))
        fine_pairs.append((human_fine, llm_fine))

        if (human_binary, human_fine) != (llm_binary, llm_fine) and len(disagreement_examples) < 15:
            disagreement_examples.append(
                {
                    "annotation_id": key[0],
                    "step_index": key[1],
                    "text_query": llm_meta[key]["text_query"],
                    "sub_question": llm_meta[key]["sub_question"],
                    "human": {
                        "entity_ambiguous": human_label[0],
                        "ambiguity_level": human_label[1],
                    },
                    "llm": {
                        "entity_ambiguous": llm_label[0],
                        "ambiguity_level": llm_label[1],
                    },
                }
            )

    binary_kappa = compute_kappa(binary_pairs, ["No", "Yes"])
    fine_kappa = compute_kappa(
        fine_pairs,
        ["No", "Object Identificattion", "Description", "Indirect Entity Ambiguity", "No Object Involved"],
    )

    return {
        "common_keys": len(common_keys),
        "paired_valid": len(binary_pairs),
        "excluded_invalid": invalid_pairs,
        "binary": {
            "kappa": binary_kappa[0],
            "observed_agreement": binary_kappa[1],
            "expected_agreement": binary_kappa[2],
            "human_counts": dict(binary_kappa[3]),
            "llm_counts": dict(binary_kappa[4]),
            "confusion": {
                f"{left}__{right}": count
                for (left, right), count in sorted(binary_kappa[5].items())
            },
        },
        "fine_grained": {
            "kappa": fine_kappa[0],
            "observed_agreement": fine_kappa[1],
            "expected_agreement": fine_kappa[2],
            "human_counts": dict(fine_kappa[3]),
            "llm_counts": dict(fine_kappa[4]),
            "confusion": {
                f"{left}__{right}": count
                for (left, right), count in sorted(fine_kappa[5].items())
            },
        },
        "disagreement_examples": disagreement_examples,
    }


def format_text(
    result: dict,
    *,
    show_confusion: bool = False,
    show_excluded: bool = False,
    show_disagreements: bool = False,
) -> str:
    lines = [
        "Entity Ambiguity Agreement Report",
        f"  Common overlapping query steps: {result['common_keys']}",
        f"  Valid paired query steps: {result['paired_valid']}",
        f"  Excluded invalid pairs: {len(result['excluded_invalid'])}",
        "",
        "Binary Kappa (Yes/No)",
        f"  kappa: {result['binary']['kappa']:.6f}",
        f"  observed_agreement: {result['binary']['observed_agreement']:.6f}",
        f"  expected_agreement: {result['binary']['expected_agreement']:.6f}",
        f"  human_counts: {json.dumps(result['binary']['human_counts'], ensure_ascii=False)}",
        f"  llm_counts: {json.dumps(result['binary']['llm_counts'], ensure_ascii=False)}",
        "",
        "Fine-grained Kappa (No + 4 ambiguity levels)",
        f"  kappa: {result['fine_grained']['kappa']:.6f}",
        f"  observed_agreement: {result['fine_grained']['observed_agreement']:.6f}",
        f"  expected_agreement: {result['fine_grained']['expected_agreement']:.6f}",
        f"  human_counts: {json.dumps(result['fine_grained']['human_counts'], ensure_ascii=False)}",
        f"  llm_counts: {json.dumps(result['fine_grained']['llm_counts'], ensure_ascii=False)}",
    ]

    if show_confusion:
        lines.extend(
            [
                "",
                "Binary confusion:",
                json.dumps(result["binary"]["confusion"], ensure_ascii=False, indent=2),
                "",
                "Fine-grained confusion:",
                json.dumps(result["fine_grained"]["confusion"], ensure_ascii=False, indent=2),
            ]
        )

    if show_excluded and result["excluded_invalid"]:
        lines.extend(
            [
                "",
                "Excluded invalid examples:",
                json.dumps(result["excluded_invalid"], ensure_ascii=False, indent=2),
            ]
        )

    if show_disagreements and result["disagreement_examples"]:
        lines.extend(
            [
                "",
                "Disagreement examples:",
                json.dumps(result["disagreement_examples"], ensure_ascii=False, indent=2),
            ]
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manual_sample = read_jsonl(args.manual_sample)
    human = read_jsonl(args.human)
    llm = read_jsonl(args.llm)

    step_sequence = build_manual_step_sequence(manual_sample)
    human_map = build_human_map(human, step_sequence)
    llm_map, llm_meta = build_llm_map(llm)
    result = summarize(manual_sample, human_map, llm_map, llm_meta)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            format_text(
                result,
                show_confusion=args.show_confusion,
                show_excluded=args.show_excluded,
                show_disagreements=args.show_disagreements,
            )
        )


if __name__ == "__main__":
    main()
