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
DEFAULT_OUTPUT = Path(
    "/home/you/FlashRAG/exps/idea10/idea_reports/docs/7-human_alignment_consistency.md"
)

TRAJECTORY_QUALITY_MAP = {
    "High: consistently useful trajectory": 3,
    "Medium: mixed progress": 2,
    "Low: mostly unhelpful trajectory": 1,
}


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
        description="Compute trajectory-level Spearman and Kendall correlation "
        "between human overall labels and TQS."
    )
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--metric", type=Path, default=DEFAULT_METRIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    human_rows = {row["sample_id"]: row for row in load_jsonl(args.human)}
    metric_rows = {row["id"]: row for row in load_jsonl(args.metric)}
    intersection_ids = sorted(set(human_rows) & set(metric_rows))

    human_scores = []
    tqs_scores = []
    for sample_id in intersection_ids:
        label = human_rows[sample_id]["trajectory_labels"]["trajectory_quality"]
        human_scores.append(TRAJECTORY_QUALITY_MAP[label])
        tqs_scores.append(float(metric_rows[sample_id]["trajectory_quality_score"]))

    spearman = spearmanr(tqs_scores, human_scores)
    kendall = kendalltau(tqs_scores, human_scores)

    text = (
        "# Human Alignment Consistency Analysis\n\n"
        f"- Spearman correlation between trajectory-level human overall labels "
        f"(mapped to `3/2/1`) and `TQS`: `rho = {spearman.statistic:.4f}` "
        f"(`p = {spearman.pvalue:.4g}`).\n"
        f"- Kendall tau between trajectory-level human overall labels "
        f"(mapped to `3/2/1`) and `TQS`: `tau = {kendall.statistic:.4f}` "
        f"(`p = {kendall.pvalue:.4g}`).\n"
    )
    args.output.write_text(text, encoding="utf-8")

    print(f"n = {len(intersection_ids)}")
    print(f"Spearman rho = {spearman.statistic:.4f} (p = {spearman.pvalue:.4g})")
    print(f"Kendall tau = {kendall.statistic:.4f} (p = {kendall.pvalue:.4g})")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
