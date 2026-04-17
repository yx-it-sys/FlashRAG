#!/usr/bin/env python3

import json
import random
from collections import defaultdict
from pathlib import Path

METRIC_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/trajectory_quality_eval/trajectory_quality_samples.jsonl"
)
LABEL_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/label/qwen/"
    "trajectory_annotation.llm_labeled.adjudicated.jsonl"
)
OUTPUT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/idea_reports/docs/"
    "6-entity_ambiguity_tqs_stratified_fairness.md"
)

BOOTSTRAP_SAMPLES = 5000
PERMUTATION_SAMPLES = 5000
RANDOM_SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iteration_bucket(num_iterations: int) -> str:
    if num_iterations <= 2:
        return "1-2"
    if num_iterations <= 4:
        return "3-4"
    return "5+"


def sample_is_ambiguous(label_row: dict) -> bool:
    for step in label_row.get("query_steps", []):
        llm_label = step.get("llm_label") or {}
        if llm_label.get("entity_ambiguous") == "Yes":
            return True
    return False


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_diff(xs: list[float], ys: list[float], rng: random.Random) -> tuple[float, float]:
    diffs = []
    for _ in range(BOOTSTRAP_SAMPLES):
        bx = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        by = [ys[rng.randrange(len(ys))] for _ in range(len(ys))]
        diffs.append(mean(by) - mean(bx))
    diffs.sort()
    low = diffs[int(0.025 * len(diffs))]
    high = diffs[int(0.975 * len(diffs))]
    return low, high


def permutation_pvalue(xs: list[float], ys: list[float], rng: random.Random) -> float:
    observed = mean(ys) - mean(xs)
    pooled = xs + ys
    n_x = len(xs)
    exceed = 0
    for _ in range(PERMUTATION_SAMPLES):
        shuffled = pooled[:]
        rng.shuffle(shuffled)
        a = shuffled[:n_x]
        b = shuffled[n_x:]
        diff = mean(b) - mean(a)
        if abs(diff) >= abs(observed):
            exceed += 1
    return (exceed + 1) / (PERMUTATION_SAMPLES + 1)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    metric_rows = {row["id"]: row for row in load_jsonl(METRIC_PATH)}
    label_rows = {row["annotation_id"]: row for row in load_jsonl(LABEL_PATH)}
    intersection_ids = sorted(set(metric_rows) & set(label_rows))

    strata: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"ambiguous": [], "non_ambiguous": []}
    )

    for sample_id in intersection_ids:
        metric = metric_rows[sample_id]
        label = label_rows[sample_id]
        key = (metric["status"], iteration_bucket(int(metric["num_iterations"])))
        target = "ambiguous" if sample_is_ambiguous(label) else "non_ambiguous"
        strata[key][target].append(float(metric["trajectory_quality_score"]))

    valid_rows = []
    pooled_weight = 0
    pooled_diff = 0.0

    for (status, bucket), groups in sorted(strata.items()):
        amb = groups["ambiguous"]
        non = groups["non_ambiguous"]
        if not amb or not non:
            continue
        row_rng = random.Random(rng.randrange(10**9))
        mean_amb = mean(amb)
        mean_non = mean(non)
        diff = mean_non - mean_amb
        ci_low, ci_high = bootstrap_diff(amb, non, row_rng)
        p_value = permutation_pvalue(amb, non, row_rng)
        weight = len(amb) + len(non)
        pooled_weight += weight
        pooled_diff += diff * weight
        valid_rows.append(
            {
                "status": status,
                "bucket": bucket,
                "n_amb": len(amb),
                "n_non": len(non),
                "mean_amb": mean_amb,
                "mean_non": mean_non,
                "diff": diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p": p_value,
                "weight": weight,
            }
        )

    weighted_diff = pooled_diff / pooled_weight if pooled_weight else float("nan")

    lines = []
    lines.append("# Stratified Fairness Analysis for Entity Ambiguity vs TQS")
    lines.append("")
    lines.append("Stratification variables:")
    lines.append("- `status ∈ {ok, missing_final_answer, generation_error}`")
    lines.append("- `num_iterations bucket ∈ {1-2, 3-4, 5+}`")
    lines.append("")
    lines.append("Within each stratum, the reported difference is `mean(non-ambiguous) - mean(ambiguous)`.")
    lines.append("")
    lines.append("| Status | Iter Bucket | Ambiguous n | Non-ambiguous n | Mean TQS (Amb) | Mean TQS (Non) | Diff | 95% Bootstrap CI | Permutation p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for row in valid_rows:
        lines.append(
            f"| `{row['status']}` | `{row['bucket']}` | {row['n_amb']} | {row['n_non']} | "
            f"{fmt(row['mean_amb'])} | {fmt(row['mean_non'])} | {fmt(row['diff'])} | "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {fmt(row['p'])} |"
        )
    lines.append("")
    lines.append("## Weighted Summary")
    lines.append("")
    lines.append(f"- Number of valid strata with both groups present: `{len(valid_rows)}`")
    lines.append(
        f"- Size-weighted average within-stratum TQS difference (`non-ambiguous - ambiguous`): `{fmt(weighted_diff)}`"
    )

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"intersection_n = {len(intersection_ids)}")
    print(f"valid_strata = {len(valid_rows)}")
    print(f"weighted_diff = {fmt(weighted_diff)}")
    for row in valid_rows:
        print(
            row["status"],
            row["bucket"],
            row["n_amb"],
            row["n_non"],
            fmt(row["diff"]),
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]",
            fmt(row["p"]),
        )
    print(f"saved_to = {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
