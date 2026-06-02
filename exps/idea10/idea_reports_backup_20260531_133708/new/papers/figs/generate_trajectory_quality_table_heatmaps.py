#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path("/home/you/FlashRAG/exps/idea10/idea_reports/papers")
RAW_DIR = BASE_DIR / "raw_data"
FIG_DIR = BASE_DIR / "figs"

TABLE_CSV = RAW_DIR / "trajectory_quality_by_task_type.csv"

TASK_ORDER = [
    "Entity Recognition",
    "Single-hop Attribute Query",
    "Multi-hop",
    "Comparison",
    "Subproblem Aggregation",
    "Macro-average",
]
TASK_SHORT = {
    "Entity Recognition": "E.R.",
    "Single-hop Attribute Query": "S.A.Q.",
    "Multi-hop": "M.H.",
    "Comparison": "Comp.",
    "Subproblem Aggregation": "S.A.",
    "Macro-average": "Avg.",
}

SETTINGS = ["Original", "Disturb", "Oracle"]
METRICS = [("rte", "RTE"), ("lj_score", "LJ score")]
SETTING_STYLE = {
    "Original": {"color": "#1f77b4", "marker": "o"},
    "Disturb": {"color": "#ff7f0e", "marker": "s"},
    "Oracle": {"color": "#2ca02c", "marker": "^"},
}


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str | None) -> float:
    if value is None or value == "" or value == "--":
        return math.nan
    return float(value)


def build_series(rows: list[dict], metric_key: str) -> dict[str, list[float]]:
    by_task_setting = {(row["task_type"], row["setting"]): row for row in rows}
    series: dict[str, list[float]] = {}
    for setting in SETTINGS:
        vals = []
        for task in TASK_ORDER:
            row = by_task_setting.get((task, setting))
            vals.append(parse_float(None if row is None else row.get(metric_key)))
        series[setting] = vals
    return series


def draw_metric_panel(ax, rows: list[dict], metric_key: str, metric_title: str):
    series = build_series(rows, metric_key)
    labels = [TASK_SHORT[t] for t in TASK_ORDER]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(metric_title, pad=10)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)

    all_vals = [v for values in series.values() for v in values if not math.isnan(v)]
    if all_vals:
        vmin, vmax = min(all_vals), max(all_vals)
        if math.isclose(vmin, vmax):
            rmin, rmax = max(0.0, vmin - 0.08), vmax + 0.08
        else:
            pad = 0.15 * (vmax - vmin)
            rmin, rmax = max(0.0, vmin - pad), vmax + pad
        ax.set_ylim(rmin, rmax)

    for setting in SETTINGS:
        style = SETTING_STYLE[setting]
        y = series[setting] + series[setting][:1]
        ax.plot(
            angles,
            y,
            label=setting,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=5.5,
        )
        ax.fill(angles, y, color=style["color"], alpha=0.08)


def save_radar_chart():
    plt.rcParams.update(
        {
            "font.size": 15,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    rows = load_rows(TABLE_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), subplot_kw={"projection": "polar"}, constrained_layout=True)
    for idx, (metric_key, metric_title) in enumerate(METRICS):
        draw_metric_panel(axes[idx], rows, metric_key, metric_title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13))

    out_path = FIG_DIR / "trajectory_quality_table_rte_lj_lines.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(str(out_path))


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_radar_chart()
