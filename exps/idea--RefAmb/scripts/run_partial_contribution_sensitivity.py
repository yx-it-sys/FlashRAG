#!/usr/bin/env python3

import argparse
import copy
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from scipy.stats import kendalltau, spearmanr


DEFAULT_METRIC = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/trajectory_quality_final_use_new_delta_f/"
    "trajectory_quality_samples.jsonl"
)
DEFAULT_HUMAN = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/label/human/TQS/human_label_150.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/"
    "trajectory_quality_final_use_new_delta_f_partial_sensitivity"
)
DEFAULT_DOC_OUTPUT = Path(
    "/home/you/FlashRAG/exps/idea10/idea_reports/docs/"
    "7-human_alignment_partial_contribution_sensitivity.md"
)
DEFAULT_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

TRAJECTORY_QUALITY_MAP = {
    "High: consistently useful trajectory": 3,
    "Medium: mixed progress": 2,
    "Low: mostly unhelpful trajectory": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the value assigned to partial_contribution, recompute TQS, "
        "and measure sensitivity against human alignment metrics."
    )
    parser.add_argument("--metric", type=Path, default=DEFAULT_METRIC)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--doc-output", type=Path, default=DEFAULT_DOC_OUTPUT)
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=DEFAULT_VALUES,
        help="Replacement values for partial_contribution utility.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def compute_cohen_kappa(human_binary: list[int], llm_binary: list[int]) -> float:
    tp = tn = fp = fn = 0
    for truth, pred in zip(human_binary, llm_binary):
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


def run_alignment_scripts(metric_path: Path, overall_md: Path, utility_md: Path) -> tuple[str, str]:
    overall_proc = subprocess.run(
        [
            sys.executable,
            "/home/you/FlashRAG/exps/idea10/scripts/compute_human_alignment_overall_corr.py",
            "--metric",
            str(metric_path),
            "--output",
            str(overall_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    utility_proc = subprocess.run(
        [
            sys.executable,
            "/home/you/FlashRAG/exps/idea10/scripts/compute_human_proxy_tqs_alignment.py",
            "--metric",
            str(metric_path),
            "--output",
            str(utility_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return overall_proc.stdout.strip(), utility_proc.stdout.strip()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.doc_output.parent.mkdir(parents=True, exist_ok=True)

    base_items = load_jsonl(args.metric)
    human_items = load_jsonl(args.human)
    human_rows = {row["sample_id"]: row for row in human_items}

    summary_rows = []
    for value in args.values:
        run_name = f"partial_{str(value).replace('.', '_')}"
        run_dir = args.output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        metric_path = run_dir / "trajectory_quality_samples.jsonl"
        overall_md = run_dir / "human_alignment_overall.md"
        utility_md = run_dir / "human_alignment_utility.md"

        items = copy.deepcopy(base_items)
        partial_step_count = 0
        for item in items:
            steps = item.get("steps", [])
            total_delta_f = 0.0
            total_step_score = 0.0
            for step in steps:
                if step.get("utility_label") == "partial_contribution":
                    step["utility"] = value
                    partial_step_count += 1
                delta_f = float(step.get("delta_f") or 0.0)
                utility = float(step.get("utility") or 0.0)
                step["step_score"] = delta_f * utility
                total_delta_f += delta_f
                total_step_score += step["step_score"]
            item["total_delta_f"] = total_delta_f
            item["total_step_score"] = total_step_score
            item["trajectory_quality_score"] = (
                statistics.mean(step["step_score"] for step in steps) if steps else 0.0
            )

        write_jsonl(metric_path, items)
        overall_stdout, utility_stdout = run_alignment_scripts(metric_path, overall_md, utility_md)

        metric_rows = {row["id"]: row for row in items}
        intersection_ids = sorted(set(human_rows) & set(metric_rows))

        human_scores = []
        tqs_scores = []
        human_binary = []
        llm_binary = []

        for sample_id in intersection_ids:
            label = human_rows[sample_id]["trajectory_labels"]["trajectory_quality"]
            human_scores.append(TRAJECTORY_QUALITY_MAP[label])
            tqs_scores.append(float(metric_rows[sample_id]["trajectory_quality_score"]))

            human_steps = {step["step_index"]: step for step in human_rows[sample_id]["steps"]}
            metric_steps = {step["iteration_index"]: step for step in metric_rows[sample_id]["steps"]}
            shared_step_indices = sorted(set(human_steps) & set(metric_steps))
            for step_index in shared_step_indices:
                usefulness = human_steps[step_index].get("usefulness")
                if usefulness is None:
                    continue
                human_binary.append(0 if usefulness == "Not useful" else 1)
                llm_binary.append(1 if float(metric_steps[step_index].get("utility") or 0.0) > 0.0 else 0)

        spearman = spearmanr(tqs_scores, human_scores)
        kendall = kendalltau(tqs_scores, human_scores)
        kappa = compute_cohen_kappa(human_binary, llm_binary)
        mean_tqs = statistics.mean(item["trajectory_quality_score"] for item in items) if items else 0.0

        summary_rows.append(
            {
                "partial_contribution_value": value,
                "partial_step_count": partial_step_count,
                "mean_tqs": mean_tqs,
                "cohen_kappa": kappa,
                "spearman": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "kendall": float(kendall.statistic),
                "kendall_p": float(kendall.pvalue),
                "metric_path": str(metric_path),
                "overall_md": str(overall_md),
                "utility_md": str(utility_md),
                "overall_stdout": overall_stdout,
                "utility_stdout": utility_stdout,
            }
        )

    summary_csv = args.output_root / "partial_contribution_sensitivity_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "partial_contribution_value",
                "partial_step_count",
                "mean_tqs",
                "cohen_kappa",
                "spearman",
                "spearman_p",
                "kendall",
                "kendall_p",
                "metric_path",
                "overall_md",
                "utility_md",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    lines = [
        "# Partial Contribution Sensitivity Analysis",
        "",
        f"Source metric file: `{args.metric}`",
        "",
        "The original directory was not modified. "
        "`partial_contribution` was swept over "
        + ", ".join(f"`{value:.1f}`" for value in args.values)
        + ".",
        "",
        "| partial_contribution | mean TQS | Cohen's Kappa | Spearman | Kendall |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['partial_contribution_value']:.1f} | {row['mean_tqs']:.6f} | "
            f"{row['cohen_kappa']:.4f} | {row['spearman']:.4f} | {row['kendall']:.4f} |"
        )
    args.doc_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved sensitivity runs to {args.output_root}")
    print(f"Saved summary CSV to {summary_csv}")
    print(f"Saved summary markdown to {args.doc_output}")
    for row in summary_rows:
        print(
            f"value={row['partial_contribution_value']:.1f} "
            f"mean_tqs={row['mean_tqs']:.6f} "
            f"kappa={row['cohen_kappa']:.4f} "
            f"spearman={row['spearman']:.4f} "
            f"kendall={row['kendall']:.4f}"
        )


if __name__ == "__main__":
    main()
