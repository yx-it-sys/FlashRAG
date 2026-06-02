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

SETTING_PATHS = {
    "Original": "trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
    "Oracle": "first_round_oracle_rewrite/trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
    "Disturb": "first_round_disturb_rewrite_static_prefix_whole/trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
}

OUT_DIR = Path("/home/you/FlashRAG/exps/idea10/idea_reports/papers/figs")
FIG_PDF = OUT_DIR / "refamb_step_score_curve_no_len_filter.pdf"
CSV_OUT = OUT_DIR / "refamb_step_score_curve_no_len_filter_means.csv"
JSON_OUT = OUT_DIR / "refamb_step_score_curve_no_len_filter_summary.json"
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 17


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_setting_samples(setting: str) -> list[dict]:
    samples: list[dict] = []
    rel_path = SETTING_PATHS[setting]
    for stage in STAGES:
        path = RESULT_ROOT / stage / rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        samples.extend(load_jsonl(path))
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


def collect_step_score_means(samples: list[dict]) -> tuple[list[float], list[float], list[float], list[int], int]:
    buckets = {i: [] for i in range(1, 6)}
    total = 0
    for row in samples:
        total += 1
        steps = row.get("steps", [])
        for i in range(1, 6):
            if len(steps) < i:
                continue
            val = steps[i - 1].get("step_score")
            if val is not None:
                buckets[i].append(float(val))

    means: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []
    counts_per_round: list[int] = []
    for i in range(1, 6):
        m, lo, hi = bootstrap_mean_ci(buckets[i])
        means.append(m)
        ci_low.append(lo)
        ci_high.append(hi)
        counts_per_round.append(len(buckets[i]))
    return means, ci_low, ci_high, counts_per_round, total


def plot_curves(
    means_by_setting: dict[str, list[float]],
    ci_low_by_setting: dict[str, list[float]],
    ci_high_by_setting: dict[str, list[float]],
) -> None:
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
    styles = {
        "Original": {"color": "#406D96", "shadow": "#A9C7E6", "marker": "o"},
        "Oracle": {"color": "#2E8B57", "shadow": "#A7D8B8", "marker": "s"},
        "Disturb": {"color": "#C46E2E", "shadow": "#F0C8A2", "marker": "^"},
    }

    for setting in ["Original", "Oracle", "Disturb"]:
        y = np.array(means_by_setting[setting], dtype=float)
        lo = np.array(ci_low_by_setting[setting], dtype=float)
        hi = np.array(ci_high_by_setting[setting], dtype=float)
        ax.fill_between(x, lo, hi, color=styles[setting]["shadow"], alpha=0.35, linewidth=0)
        ax.plot(
            x,
            y,
            color=styles[setting]["color"],
            marker=styles[setting]["marker"],
            markersize=9,
            linewidth=3,
            label=setting,
        )

    ax.set_xlabel("Iteration Round", fontsize=24, fontweight="bold")
    ax.set_ylabel("Mean $S_t$", fontsize=24, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.85, 5.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="upper right", fontsize=22)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(22)
        label.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(FIG_PDF)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    means_by_setting: dict[str, list[float]] = {}
    ci_low_by_setting: dict[str, list[float]] = {}
    ci_high_by_setting: dict[str, list[float]] = {}
    counts: dict[str, dict[str, object]] = {}

    for setting in ["Original", "Oracle", "Disturb"]:
        samples = load_setting_samples(setting)
        means, ci_low, ci_high, counts_per_round, total = collect_step_score_means(samples)
        means_by_setting[setting] = means
        ci_low_by_setting[setting] = ci_low
        ci_high_by_setting[setting] = ci_high
        counts[setting] = {
            "total_items": total,
            "n_items_with_this_round": counts_per_round,
        }

    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "round", "mean_step_score", "ci_low", "ci_high", "n_items_with_this_round"])
        for setting in ["Original", "Oracle", "Disturb"]:
            for i in range(5):
                writer.writerow([
                    setting,
                    i + 1,
                    f"{means_by_setting[setting][i]:.6f}",
                    f"{ci_low_by_setting[setting][i]:.6f}",
                    f"{ci_high_by_setting[setting][i]:.6f}",
                    counts[setting]["n_items_with_this_round"][i],
                ])

    plot_curves(means_by_setting, ci_low_by_setting, ci_high_by_setting)

    payload = {
        "means_by_setting": means_by_setting,
        "ci_low_by_setting": ci_low_by_setting,
        "ci_high_by_setting": ci_high_by_setting,
        "counts": counts,
        "figure_pdf": str(FIG_PDF),
        "csv": str(CSV_OUT),
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
