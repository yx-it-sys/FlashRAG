#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from scipy.stats import kendalltau, spearmanr

DEFAULT_HUMAN = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/label/human/TQS/human_label_150.jsonl"
)
DEFAULT_METRIC = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/trajectory_quality_eval/trajectory_quality_samples.jsonl"
)

NOVELTY_MAP = {
    "No new information": 0.0,
    "Partial new information": 0.5,
    "Clear new information": 1.0,
}

USEFULNESS_MAP = {
    "Not useful": 0.0,
    "Partly useful": 0.5,
    "Clearly useful": 1.0,
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_step_means(human_row: dict, metric_row: dict) -> dict:
    novelty_h = []
    usefulness_h = []
    step_proxy_h = []
    delta_f = []
    utility = []
    step_score = []

    human_steps = {step["step_index"]: step for step in human_row["steps"]}
    metric_steps = {step["iteration_index"]: step for step in metric_row["steps"]}
    shared_step_indices = sorted(set(human_steps) & set(metric_steps))

    for step_index in shared_step_indices:
        h_step = human_steps[step_index]
        m_step = metric_steps[step_index]

        novelty_label = h_step.get("new_information")
        usefulness_label = h_step.get("usefulness")
        redundant_label = h_step.get("redundant")

        if (
            h_step["step_index"] != 1
            and novelty_label is None
            and usefulness_label is None
            and redundant_label is None
        ):
            continue

        if novelty_label is None or usefulness_label is None:
            raise ValueError(
                f"Partial missing human step label for {human_row['sample_id']} "
                f"step {h_step['step_index']}"
            )

        novelty_value = NOVELTY_MAP[novelty_label]
        usefulness_value = USEFULNESS_MAP[usefulness_label]
        novelty_h.append(novelty_value)
        usefulness_h.append(usefulness_value)
        step_proxy_h.append(novelty_value * usefulness_value)

        delta_f.append(float(m_step["delta_f"]))
        utility.append(float(m_step["utility"]))
        step_score.append(float(m_step["step_score"]))

    return {
        "mean_h_novelty": mean(novelty_h),
        "mean_h_usefulness": mean(usefulness_h),
        "human_proxy_tqs": mean(step_proxy_h),
        "mean_delta_f": mean(delta_f),
        "mean_utility": mean(utility),
        "mean_step_score": mean(step_score),
        "num_effective_steps": len(step_proxy_h),
        "num_shared_steps": len(shared_step_indices),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute human-proxy TQS alignment from step-level novelty/usefulness labels."
    )
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--metric", type=Path, default=DEFAULT_METRIC)
    return parser.parse_args()


def report_pair(name_a: str, xs: list[float], name_b: str, ys: list[float]) -> None:
    spearman = spearmanr(xs, ys)
    kendall = kendalltau(xs, ys)
    print(f"{name_a} vs {name_b}")
    print(f"  Spearman rho = {spearman.statistic:.4f} (p = {spearman.pvalue:.4g})")
    print(f"  Kendall tau = {kendall.statistic:.4f} (p = {kendall.pvalue:.4g})")


def main() -> None:
    args = parse_args()
    human_rows = {row["sample_id"]: row for row in load_jsonl(args.human)}
    metric_rows = {row["id"]: row for row in load_jsonl(args.metric)}
    intersection_ids = sorted(set(human_rows) & set(metric_rows))

    mean_h_novelty = []
    mean_h_usefulness = []
    human_proxy_tqs = []
    mean_delta_f = []
    mean_utility = []
    tqs = []
    mean_step_score = []
    skipped_no_shared_steps = []

    for sample_id in intersection_ids:
        stats = compute_step_means(human_rows[sample_id], metric_rows[sample_id])
        if stats["num_shared_steps"] == 0:
            skipped_no_shared_steps.append(sample_id)
            continue
        mean_h_novelty.append(stats["mean_h_novelty"])
        mean_h_usefulness.append(stats["mean_h_usefulness"])
        human_proxy_tqs.append(stats["human_proxy_tqs"])
        mean_delta_f.append(stats["mean_delta_f"])
        mean_utility.append(stats["mean_utility"])
        mean_step_score.append(stats["mean_step_score"])
        tqs.append(float(metric_rows[sample_id]["trajectory_quality_score"]))

    print(f"n = {len(tqs)}")
    if skipped_no_shared_steps:
        print(
            f"skipped_no_shared_steps = {len(skipped_no_shared_steps)}: "
            + ", ".join(skipped_no_shared_steps)
        )
    report_pair("mean_delta_f", mean_delta_f, "mean_h_novelty", mean_h_novelty)
    report_pair("mean_utility", mean_utility, "mean_h_usefulness", mean_h_usefulness)
    report_pair("TQS", tqs, "human_proxy_tqs", human_proxy_tqs)
    report_pair("mean_step_score", mean_step_score, "human_proxy_tqs", human_proxy_tqs)


if __name__ == "__main__":
    main()
