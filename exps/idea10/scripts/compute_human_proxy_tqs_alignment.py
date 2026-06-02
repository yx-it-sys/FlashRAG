#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


DEFAULT_HUMAN = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/label/human/TQS/human_label_150.jsonl"
)
DEFAULT_METRIC = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/trajectory_quality_final_use_new_delta_f/trajectory_quality_samples.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/home/you/FlashRAG/exps/idea10/idea_reports/docs/7-human_alignment_consistency.md"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute binary alignment between human usefulness labels and "
        "LLM utility on all shared steps."
    )
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--metric", type=Path, default=DEFAULT_METRIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def human_usefulness_to_binary(label: str) -> int:
    if label == "Not useful":
        return 0
    if label in {"Partly useful", "Clearly useful"}:
        return 1
    raise ValueError(f"Unsupported human usefulness label: {label!r}")


def llm_utility_to_binary(value: float) -> int:
    return 1 if float(value) > 0.0 else 0


def matthews_corrcoef_binary(y_true: list[int], y_pred: list[int]) -> float:
    tp = tn = fp = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 1 and pred == 0:
            fn += 1

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0.0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def cohens_kappa_binary(y_true: list[int], y_pred: list[int]) -> float:
    tp = tn = fp = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

    n = tp + tn + fp + fn
    if n == 0:
        return 0.0

    p_o = (tp + tn) / n
    p_yes_h = (tp + fn) / n
    p_no_h = (tn + fp) / n
    p_yes_l = (tp + fp) / n
    p_no_l = (tn + fn) / n
    p_e = p_yes_h * p_yes_l + p_no_h * p_no_l
    denom = 1.0 - p_e
    if denom == 0.0:
        return 0.0
    return (p_o - p_e) / denom


def main() -> None:
    args = parse_args()
    human_rows = {row["sample_id"]: row for row in load_jsonl(args.human)}
    metric_rows = {row["id"]: row for row in load_jsonl(args.metric)}
    intersection_ids = sorted(set(human_rows) & set(metric_rows))

    human_binary = []
    llm_binary = []

    tp = tn = fp = fn = 0
    total_shared_steps = 0
    total_evaluated_steps = 0

    for sample_id in intersection_ids:
        human_steps = {step["step_index"]: step for step in human_rows[sample_id]["steps"]}
        metric_steps = {step["iteration_index"]: step for step in metric_rows[sample_id]["steps"]}
        shared_step_indices = sorted(set(human_steps) & set(metric_steps))
        total_shared_steps += len(shared_step_indices)

        for step_index in shared_step_indices:
            h_step = human_steps[step_index]
            m_step = metric_steps[step_index]

            usefulness_label = h_step.get("usefulness")
            if usefulness_label is None:
                continue

            h_bin = human_usefulness_to_binary(usefulness_label)
            m_bin = llm_utility_to_binary(float(m_step.get("utility") or 0.0))

            human_binary.append(h_bin)
            llm_binary.append(m_bin)
            total_evaluated_steps += 1

            if h_bin == 1 and m_bin == 1:
                tp += 1
            elif h_bin == 0 and m_bin == 0:
                tn += 1
            elif h_bin == 0 and m_bin == 1:
                fp += 1
            else:
                fn += 1

    mcc = matthews_corrcoef_binary(human_binary, llm_binary)
    kappa = cohens_kappa_binary(human_binary, llm_binary)
    accuracy = (tp + tn) / total_evaluated_steps if total_evaluated_steps else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print(f"sample_intersection = {len(intersection_ids)}")
    print(f"total_shared_steps = {total_shared_steps}")
    print(f"total_evaluated_steps = {total_evaluated_steps}")
    print("human usefulness mapping: Not useful -> 0; Partly/Clearly useful -> 1")
    print("llm utility mapping: utility == 0 -> 0; utility > 0 -> 1")
    print(f"Cohen's kappa = {kappa:.4f}")
    print(f"phi/MCC = {mcc:.4f}")
    print(f"accuracy = {accuracy:.4f}")
    print(f"precision = {precision:.4f}")
    print(f"recall = {recall:.4f}")
    print(f"f1 = {f1:.4f}")
    print(f"confusion_matrix = [[tn={tn}, fp={fp}], [fn={fn}, tp={tp}]]")

    markdown = (
        "\n\n## Binary Human-LLM Utility Alignment (All Shared Steps)\n\n"
        "- Human utility source: `human_label_150.jsonl` field `steps[*].usefulness`\n"
        "- Human binarization: `Not useful -> 0`, `Partly useful / Clearly useful -> 1`\n"
        "- LLM binarization: `utility == 0 -> 0`, `utility > 0 -> 1`\n"
        "- Step filtering: all shared steps with human usefulness labels\n\n"
        f"- Sample intersection: `{len(intersection_ids)}`\n"
        f"- Total shared steps: `{total_shared_steps}`\n"
        f"- Total evaluated steps: `{total_evaluated_steps}`\n\n"
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        f"| Cohen's Kappa | {kappa:.4f} |\n"
        f"| Phi / MCC | {mcc:.4f} |\n"
        f"| Accuracy | {accuracy:.4f} |\n"
        f"| Precision | {precision:.4f} |\n"
        f"| Recall | {recall:.4f} |\n"
        f"| F1 | {f1:.4f} |\n\n"
        "| Confusion Matrix | Count |\n"
        "| --- | ---: |\n"
        f"| TN | {tn} |\n"
        f"| FP | {fp} |\n"
        f"| FN | {fn} |\n"
        f"| TP | {tp} |\n"
    )
    with args.output.open("a", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Appended markdown to {args.output}")


if __name__ == "__main__":
    main()
