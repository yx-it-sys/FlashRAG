#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT_ROOT = Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B")
STAGES = [
    "RefAmb_2026_05_01_11_26_refamb_mcsearch_stage",
    "RefAmb_2026_05_01_13_28_refamb_oven_stage",
    "RefAmb_2026_05_02_13_50_refamb_infoseek_stage",
    "RefAmb_2026_05_02_14_22_refamb_crag_stage",
]

METRIC_REL_PATH = "trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl"
LABEL_REL_PATH = "label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.jsonl"

OUT_DIR = Path("/home/you/FlashRAG/exps/idea10/idea_reports/papers/figs")
FIG_PDF = OUT_DIR / "refamb_step_score_curve_ambiguity_split.pdf"
FIG_PNG = OUT_DIR / "refamb_step_score_curve_ambiguity_split.png"
CSV_OUT = OUT_DIR / "refamb_step_score_curve_ambiguity_split_means.csv"
JSON_OUT = OUT_DIR / "refamb_step_score_curve_ambiguity_split_summary.json"

MAX_ROUNDS = 5
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 17

GROUP_ORDER = ["Ambiguous", "Non-ambiguous"]
GROUP_LABELS = {
    "Ambiguous": "Entity ambiguous",
    "Non-ambiguous": "Entity non-ambiguous",
}
GROUP_STYLES = {
    "Ambiguous": {"color": "#406D96", "shadow": "#A9C7E6", "marker": "o"},
    "Non-ambiguous": {"color": "#C46E2E", "shadow": "#F0C8A2", "marker": "s"},
}


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def infer_entity_ambiguity(label_row: dict) -> str | None:
    labels = []
    for step in label_row.get("trajectory") or []:
        if not isinstance(step, dict):
            continue
        llm_label = step.get("llm_label")
        if isinstance(llm_label, dict):
            value = llm_label.get("entity_ambiguous")
            if value in {"Yes", "No"}:
                labels.append(value)
    if "Yes" in labels:
        return "Ambiguous"
    if "No" in labels:
        return "Non-ambiguous"
    return None


def load_original_samples() -> list[dict]:
    samples: list[dict] = []
    for stage in STAGES:
        metric_path = RESULT_ROOT / stage / METRIC_REL_PATH
        label_path = RESULT_ROOT / stage / LABEL_REL_PATH
        if not metric_path.exists():
            raise FileNotFoundError(metric_path)
        if not label_path.exists():
            raise FileNotFoundError(label_path)

        metrics = {row["id"]: row for row in load_jsonl(metric_path)}
        labels = {row["id"]: row for row in load_jsonl(label_path)}

        common_ids = sorted(metrics.keys() & labels.keys())
        for sample_id in common_ids:
            metric_row = metrics[sample_id]
            group = infer_entity_ambiguity(labels[sample_id])
            if group is None:
                continue
            samples.append(
                {
                    "id": sample_id,
                    "stage": stage,
                    "group": group,
                    "steps": metric_row.get("steps", []),
                }
            )
    return samples


def bootstrap_mean_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean_value = float(arr.mean())
    if arr.size == 1:
        return mean_value, mean_value, mean_value
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(BOOTSTRAP_SAMPLES, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return (
        mean_value,
        float(np.quantile(boot_means, 0.025)),
        float(np.quantile(boot_means, 0.975)),
    )


def collect_group_curve_stats(samples: list[dict]) -> dict[str, dict[str, object]]:
    grouped = {group: [] for group in GROUP_ORDER}
    for row in samples:
        grouped[row["group"]].append(row)

    stats: dict[str, dict[str, object]] = {}
    for group in GROUP_ORDER:
        rows = grouped[group]
        padded_values_by_round: list[list[float]] = [[] for _ in range(MAX_ROUNDS)]
        active_values_by_round: list[list[float]] = [[] for _ in range(MAX_ROUNDS)]
        active_counts = [0 for _ in range(MAX_ROUNDS)]

        for row in rows:
            steps = row.get("steps", [])
            for round_idx in range(MAX_ROUNDS):
                value = 0.0
                if round_idx < len(steps):
                    raw_value = steps[round_idx].get("step_score")
                    if raw_value is not None:
                        value = float(raw_value)
                        active_counts[round_idx] += 1
                        active_values_by_round[round_idx].append(value)
                padded_values_by_round[round_idx].append(value)

        padded_means: list[float] = []
        padded_ci_low: list[float] = []
        padded_ci_high: list[float] = []
        conditional_means: list[float] = []
        conditional_ci_low: list[float] = []
        conditional_ci_high: list[float] = []
        for round_idx in range(MAX_ROUNDS):
            mean, lo, hi = bootstrap_mean_ci(padded_values_by_round[round_idx])
            padded_means.append(mean)
            padded_ci_low.append(lo)
            padded_ci_high.append(hi)

            c_mean, c_lo, c_hi = bootstrap_mean_ci(active_values_by_round[round_idx])
            conditional_means.append(c_mean)
            conditional_ci_low.append(c_lo)
            conditional_ci_high.append(c_hi)

        stats[group] = {
            "sample_count": len(rows),
            "active_counts": active_counts,
            "padded_values_by_round": padded_values_by_round,
            "active_values_by_round": active_values_by_round,
            "padded_means": padded_means,
            "padded_ci_low": padded_ci_low,
            "padded_ci_high": padded_ci_high,
            "conditional_means": conditional_means,
            "conditional_ci_low": conditional_ci_low,
            "conditional_ci_high": conditional_ci_high,
        }
    return stats


def plot_curves(stats: dict[str, dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "mathtext.fontset": "dejavuserif",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.edgecolor": "black",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 7.6))
    x = np.array([1, 2, 3, 4, 5], dtype=float)

    for group in GROUP_ORDER:
        group_stats = stats[group]
        y = np.array(group_stats["padded_means"], dtype=float)
        lo = np.array(group_stats["padded_ci_low"], dtype=float)
        hi = np.array(group_stats["padded_ci_high"], dtype=float)
        style = GROUP_STYLES[group]
        label = f"{GROUP_LABELS[group]} (n={group_stats['sample_count']})"

        ax.fill_between(x, lo, hi, color=style["shadow"], alpha=0.35, linewidth=0)
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            markersize=9,
            linewidth=3,
            label=label,
        )

    ax.set_xlabel("Iteration Round", fontsize=24, fontweight="bold")
    ax.set_ylabel("Zero-padded mean $S_t$", fontsize=24, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.85, 5.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="upper right", fontsize=20)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(22)
        label.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_original_samples()
    stats = collect_group_curve_stats(samples)

    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "round",
                "padded_mean",
                "padded_ci_low",
                "padded_ci_high",
                "conditional_mean",
                "conditional_ci_low",
                "conditional_ci_high",
                "active_count",
                "total_count",
            ]
        )
        for group in GROUP_ORDER:
            group_stats = stats[group]
            total_count = group_stats["sample_count"]
            for round_idx in range(MAX_ROUNDS):
                writer.writerow(
                    [
                        group,
                        round_idx + 1,
                        f"{group_stats['padded_means'][round_idx]:.6f}",
                        f"{group_stats['padded_ci_low'][round_idx]:.6f}",
                        f"{group_stats['padded_ci_high'][round_idx]:.6f}",
                        f"{group_stats['conditional_means'][round_idx]:.6f}",
                        f"{group_stats['conditional_ci_low'][round_idx]:.6f}",
                        f"{group_stats['conditional_ci_high'][round_idx]:.6f}",
                        group_stats["active_counts"][round_idx],
                        total_count,
                    ]
                )

    plot_curves(stats)

    summary = {
        "figure_pdf": str(FIG_PDF),
        "figure_png": str(FIG_PNG),
        "csv": str(CSV_OUT),
        "groups": {
            group: {
                "sample_count": stats[group]["sample_count"],
                "active_counts": stats[group]["active_counts"],
                "padded_means": stats[group]["padded_means"],
                "padded_ci_low": stats[group]["padded_ci_low"],
                "padded_ci_high": stats[group]["padded_ci_high"],
                "conditional_means": stats[group]["conditional_means"],
                "conditional_ci_low": stats[group]["conditional_ci_low"],
                "conditional_ci_high": stats[group]["conditional_ci_high"],
            }
            for group in GROUP_ORDER
        },
    }
    JSON_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
