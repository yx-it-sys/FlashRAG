#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "figs" / "interventions" / "decay_contribution_decomposition"
OUT_PREFIX = "decay_contribution_decomposition_heatmap_v1"

MODELS = ["Qwen2.5-vl-7B", "Qwen3-vl-4B", "Qwen3-vl-8B", "Qwen3-vl-32B", "InternVL3.5-8B"]
SETTINGS = ["Original", "Disturb", "Oracle"]
FACTORS = ["C_surv", "C_cond"]
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
            payload[model][setting] = {factor: {} for factor in FACTORS}
            for t in STEPS:
                st = zero_pad[(model, setting, t)]
                mut = alive[(model, setting, t)]
                denom = abs(np.log(st / s1)) if st > 0 and s1 > 0 else np.nan
                c_surv = abs(np.log(p_surv[t] / p_surv[1])) / denom if denom and not np.isnan(denom) else np.nan
                c_cond = abs(np.log(mut / mu1)) / denom if denom and not np.isnan(denom) else np.nan
                payload[model][setting]["C_surv"][t] = float(c_surv)
                payload[model][setting]["C_cond"][t] = float(c_cond)
    return payload


def make_heatmap(ax, data: np.ndarray, title: str, vmin: float, vmax: float, show_ylabels: bool):
    cmap = plt.cm.YlOrRd
    im = ax.imshow(data, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    ax.set_xticks(np.arange(len(STEPS)))
    ax.set_xticklabels([str(t) for t in STEPS], fontsize=10)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS if show_ylabels else [""] * len(MODELS), fontsize=10)
    ax.tick_params(axis="both", length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                txt = "n/a"
                color = "black"
            else:
                txt = f"{val:.2f}"
                color = "white" if val >= 0.95 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, fontweight="bold", color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_series()

    vmax = 0.0
    for model in MODELS:
        for setting in SETTINGS:
            for factor in FACTORS:
                for t in STEPS:
                    val = payload[model][setting][factor][t]
                    if not np.isnan(val):
                        vmax = max(vmax, val)
    vmax = max(vmax, 1.8)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "font.weight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "mathtext.fontset": "dejavuserif",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.pad_inches": 0.05,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=False)
    factors_titles = [r"$C_{\mathrm{surv}}$", r"$C_{\mathrm{cond}}$"]

    ims = []
    for row_idx, factor in enumerate(FACTORS):
        for col_idx, setting in enumerate(SETTINGS):
            ax = axes[row_idx, col_idx]
            data = np.array([[payload[m][setting][factor][t] for t in STEPS] for m in MODELS], dtype=float)
            im = make_heatmap(ax, data, f"{setting} | {factors_titles[row_idx]}", 0.0, vmax, show_ylabels=(col_idx == 0))
            ims.append(im)
            if row_idx == 1:
                ax.set_xlabel("t", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("Model", fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("")

    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Contribution ratio", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Log-scale decomposition of zero-padded trajectory decay",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.11, right=0.95, top=0.92, bottom=0.08, wspace=0.08, hspace=0.22)

    pdf_path = OUT_DIR / f"{OUT_PREFIX}.pdf"
    png_path = OUT_DIR / f"{OUT_PREFIX}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
