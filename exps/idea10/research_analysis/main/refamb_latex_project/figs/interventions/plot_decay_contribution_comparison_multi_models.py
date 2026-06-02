#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "figs" / "interventions" / "decay_contribution_decomposition"
OUT_PREFIX = "decay_contribution_comparison_v3_transposed"

MODELS = ["Qwen2.5-vl-7B", "Qwen3-vl-4B", "Qwen3-vl-8B", "Qwen3-vl-32B", "InternVL3.5-8B"]
SETTINGS = ["Original", "Disturb", "Oracle"]
STEPS = [2, 3, 4, 5]


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_series() -> dict[str, dict[str, dict[str, dict[int, float]]]]:
    base = PROJECT_ROOT / "figs" / "interventions"
    zero_pad_csv = base / "true_trajectory_decay_zero_pad" / "true_trajectory_step_score_decay_scheme1_compact_multi_v6_means.csv"
    alive_csv = base / "true_trajectory_decay_alive_only" / "true_trajectory_step_score_decay_alive_only_v3_summary.csv"
    length_csv = base / "trajectory_length_distribution" / "trajectory_length_distribution_full_universe_multi_v1.csv"

    zero_pad = {}
    for row in read_csv_rows(zero_pad_csv):
        zero_pad[(row["model"], row["setting"], int(row["step"]))] = float(row["mean_step_score"])

    alive = {}
    for row in read_csv_rows(alive_csv):
        alive[(row["model"], row["setting"], int(row["step"]))] = float(row["mean_step_score"])

    length_probs: dict[tuple[str, str], dict[int, float]] = {}
    for row in read_csv_rows(length_csv):
        key = (row["model"], row["setting"])
        length_probs.setdefault(key, {})[int(row["length"])] = float(row["probability"])

    payload: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    for model in MODELS:
        payload[model] = {}
        for setting in SETTINGS:
            probs = length_probs[(model, setting)]
            p_surv = {t: sum(v for l, v in probs.items() if l >= t) for t in range(1, 6)}
            s1 = zero_pad[(model, setting, 1)]
            mu1 = alive[(model, setting, 1)]
            payload[model][setting] = {"C_surv": {}, "C_cond": {}}
            for t in STEPS:
                st = zero_pad[(model, setting, t)]
                mut = alive[(model, setting, t)]
                denom = abs(np.log(st / s1)) if st > 0 and s1 > 0 else np.nan
                c_surv = abs(np.log(p_surv[t] / p_surv[1])) / denom if denom and not np.isnan(denom) else np.nan
                c_cond = abs(np.log(mut / mu1)) / denom if denom and not np.isnan(denom) else np.nan
                payload[model][setting]["C_surv"][t] = float(c_surv)
                payload[model][setting]["C_cond"][t] = float(c_cond)
    return payload


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_series()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "font.weight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "mathtext.fontset": "dejavuserif",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.pad_inches": 0.05,
        }
    )

    fig = plt.figure(figsize=(17.2, 8.4))
    gs = GridSpec(3, 5, figure=fig, wspace=0.16, hspace=0.22)
    colors = {"C_surv": "#C46E2E", "C_cond": "#2E8B57"}
    styles = {"C_surv": "--", "C_cond": "-"}
    markers = {"C_surv": "o", "C_cond": "s"}

    axes = []
    for i, setting in enumerate(SETTINGS):
        row_axes = []
        for j, model in enumerate(MODELS):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
            x = np.array(STEPS, dtype=float)
            y_surv = np.array([payload[model][setting]["C_surv"][t] for t in STEPS], dtype=float)
            y_cond = np.array([payload[model][setting]["C_cond"][t] for t in STEPS], dtype=float)

            ax.plot(x, y_surv, color=colors["C_surv"], linestyle=styles["C_surv"], marker=markers["C_surv"], linewidth=1.9, markersize=5.0, label=r"$C_{\mathrm{surv}}$")
            ax.plot(x, y_cond, color=colors["C_cond"], linestyle=styles["C_cond"], marker=markers["C_cond"], linewidth=1.9, markersize=5.0, label=r"$C_{\mathrm{cond}}$")
            if i == 0:
                ax.set_title(model, fontsize=10.5, fontweight="bold", pad=4)
            ax.set_xticks(STEPS)
            ax.set_ylim(0, 1.85)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(f"{setting}\nContribution ratio", fontsize=10, fontweight="bold")
            if i == len(SETTINGS) - 1:
                ax.set_xlabel("t", fontsize=10, fontweight="bold")
            ax.legend(
                loc="upper right",
                frameon=True,
                framealpha=0.92,
                fontsize=8.5,
                borderaxespad=0.25,
                handlelength=1.6,
                labelspacing=0.22,
            )
        axes.append(row_axes)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.975, bottom=0.08)

    pdf_path = OUT_DIR / f"{OUT_PREFIX}.pdf"
    png_path = OUT_DIR / f"{OUT_PREFIX}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
