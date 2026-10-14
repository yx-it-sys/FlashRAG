from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


PROJECT_ROOT = Path("/home/you/FlashRAG/exps/idea10")
DEFAULT_DATA_PATH = PROJECT_ROOT / "data/datasets/RefAmb/original_balanced.jsonl"
DEFAULT_OUT_PATH = PROJECT_ROOT / "idea_reports/refamb_analysis_figs/refamb_original_balanced_source_sankey.png"

SOURCE_DISPLAY_NAMES = {
    "infoseek": "InfoSeek",
    "mcsearch": "MC-Search",
    "crag": "CRAG-MM",
    "oven": "OVEN",
}

SOURCE_COLORS = {
    "infoseek": "#88B2DC",
    "mcsearch": "#99D0C9",
    "crag": "#F0AE67",
    "oven": "#BE8CB8",
}

TASK_COLORS = {
    "Single-hop Attribute Query": "#8fb3d5",
    "Entity Recognition": "#c2d3de",
    "Subproblem Aggregation": "#b9ddd7",
    "Multi-hop": "#efc8a8",
    "Comparison": "#cdb9d3",
}

TASK_LABELS = [
    "Single-hop Attribute Query",
    "Entity Recognition",
    "Subproblem Aggregation",
    "Multi-hop",
    "Comparison",
]

TASK_DISPLAY_NAMES = {
    "Single-hop Attribute Query": "S.A.Q.",
    "Entity Recognition": "E.R.",
    "Subproblem Aggregation": "C.C.Q.",
    "Multi-hop": "M.H.",
    "Comparison": "Comp.",
}

NODE_EDGE = (0.55, 0.55, 0.55, 0.9)
TEXT_COLOR = "black"


def load_counts(path: Path) -> tuple[Counter, Counter, Counter]:
    source_counts = Counter()
    task_counts = Counter()
    pair_counts = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source = row["source"]
            task = row["task_type"]
            source_counts[source] += 1
            task_counts[task] += 1
            pair_counts[(source, task)] += 1
    return source_counts, task_counts, pair_counts


def ordered_sources(counts: Counter) -> list[str]:
    preferred_order = ["infoseek", "mcsearch", "crag", "oven"]
    ordered = [name for name in preferred_order if name in counts]
    ordered.extend(sorted(name for name in counts if name not in preferred_order))
    return ordered


def apply_serif_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 0.8,
        }
    )


def draw_pie(ax, counts: Counter, fontsize: int):
    labels = ordered_sources(counts)
    values = [counts[name] for name in labels]
    display_labels = [SOURCE_DISPLAY_NAMES.get(name, name) for name in labels]

    ax.set_facecolor("white")
    wedges, _ = ax.pie(
        values,
        startangle=90,
        counterclock=False,
        radius=1.16,
        wedgeprops=dict(linewidth=1.2, edgecolor="white"),
        colors=[SOURCE_COLORS[name] for name in labels],
    )

    total = sum(values)
    for wedge, label, value in zip(wedges, display_labels, values):
        theta = (wedge.theta1 + wedge.theta2) / 2.0
        theta_rad = theta * 3.141592653589793 / 180.0
        span = abs(wedge.theta2 - wedge.theta1)
        mid_radius = 0.60 if span >= 80 else 0.56 if span >= 50 else 0.52
        x = mid_radius * math.cos(theta_rad)
        y = mid_radius * math.sin(theta_rad)
        if label == "OVEN":
            y += 0.06
        ax.text(
            x,
            y + 0.045,
            label,
            ha="center",
            va="center",
            fontsize=max(fontsize - 2, 11),
            family="serif",
            color=TEXT_COLOR,
            weight="bold",
        )
        ax.text(
            x,
            y - 0.045,
            f"{value / total * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=max(fontsize - 4, 9),
            family="serif",
            color=TEXT_COLOR,
            weight="bold",
        )

    ax.set_aspect("equal")
    ax.set_axis_off()


def build_layout(counts: Counter, order: list[str], top: float, bottom: float, gap: float = 0.028):
    total = sum(counts[k] for k in order)
    available = top - bottom - gap * (len(order) - 1)
    scale = available / total
    positions = {}
    cursor = top
    for key in order:
        h = counts[key] * scale
        positions[key] = (cursor - h, cursor)
        cursor = cursor - h - gap
    return positions


def blend_with_white(hex_color: str, mix: float = 0.12, alpha: float = 0.42):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    r = r + (1.0 - r) * mix
    g = g + (1.0 - g) * mix
    b = b + (1.0 - b) * mix
    return (r, g, b, alpha)


def add_ribbon(ax, x0, x1, y0_bottom, y0_top, y1_bottom, y1_top, color):
    dx = x1 - x0
    c = dx * 0.42
    verts = [
        (x0, y0_top),
        (x0 + c, y0_top),
        (x1 - c, y1_top),
        (x1, y1_top),
        (x1, y1_bottom),
        (x1 - c, y1_bottom),
        (x0 + c, y0_bottom),
        (x0, y0_bottom),
        (x0, y0_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none"))


def task_label_display(task: str) -> str:
    return TASK_DISPLAY_NAMES.get(task, task.replace(" ", "\n"))


def draw_sankey(ax, source_counts: Counter, task_counts: Counter, pair_counts: Counter, fontsize: int):
    ax.set_facecolor("white")

    left_x0, left_x1 = 0.14, 0.24
    right_x0, right_x1 = 0.72, 0.82
    top, bottom = 0.88, 0.16

    source_order = ordered_sources(source_counts)
    task_order = [task for task in TASK_LABELS if task in task_counts]

    source_pos = build_layout(source_counts, source_order, top=top, bottom=bottom, gap=0.035)
    task_pos = build_layout(task_counts, task_order, top=top, bottom=bottom, gap=0.035)

    source_offsets = {k: source_pos[k][0] for k in source_order}
    task_offsets = {k: task_pos[k][0] for k in task_order}
    ribbon_colors = {k: blend_with_white(v) for k, v in SOURCE_COLORS.items()}

    for source_key in source_order:
        for task in task_order:
            value = pair_counts.get((source_key, task), 0)
            if value <= 0:
                continue
            sy0 = source_offsets[source_key]
            sy1 = sy0 + value * (source_pos[source_key][1] - source_pos[source_key][0]) / source_counts[source_key]
            ty0 = task_offsets[task]
            ty1 = ty0 + value * (task_pos[task][1] - task_pos[task][0]) / task_counts[task]
            add_ribbon(ax, left_x1, right_x0, sy0, sy1, ty0, ty1, ribbon_colors[source_key])
            source_offsets[source_key] = sy1
            task_offsets[task] = ty1

    node_width = left_x1 - left_x0
    right_node_width = right_x1 - right_x0
    for source_key in source_order:
        y0, y1 = source_pos[source_key]
        ax.add_patch(Rectangle((left_x0, y0), node_width, y1 - y0, facecolor=SOURCE_COLORS[source_key], edgecolor=NODE_EDGE, linewidth=0.8))
        ax.text(
            left_x0,
            (y0 + y1) / 2,
            SOURCE_DISPLAY_NAMES.get(source_key, source_key),
            ha="left",
            va="center",
            fontsize=fontsize,
            family="serif",
            color=TEXT_COLOR,
            weight="bold",
        )

    for task in task_order:
        y0, y1 = task_pos[task]
        ax.add_patch(Rectangle((right_x0, y0), right_node_width, y1 - y0, facecolor=TASK_COLORS[task], edgecolor="none", linewidth=0.0))
        ax.text(
            right_x0 + right_node_width / 2,
            (y0 + y1) / 2,
            task_label_display(task),
            ha="center",
            va="center",
            fontsize=fontsize,
            family="serif",
            color=TEXT_COLOR,
            weight="bold",
        )

    ax.text(left_x0, top + 0.02, "Source", ha="left", va="bottom", fontsize=fontsize, family="serif", weight="bold", color=TEXT_COLOR)
    ax.text(right_x1, top + 0.02, "Task Type", ha="right", va="bottom", fontsize=fontsize, family="serif", weight="bold", color=TEXT_COLOR)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()


def build_pie_figure(counts: Counter, fontsize: int):
    apply_serif_style()
    fig, ax = plt.subplots(figsize=(8.8, 6.8), facecolor="white")
    draw_pie(ax, counts, fontsize)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.10)
    fig.text(
        0.5,
        0.02,
        "(a) RefAmb source distribution",
        ha="center",
        va="bottom",
        fontsize=max(fontsize - 1, 12),
        family="serif",
        weight="bold",
        color=TEXT_COLOR,
    )
    return fig


def build_sankey_figure(source_counts: Counter, task_counts: Counter, pair_counts: Counter, fontsize: int):
    apply_serif_style()
    fig = plt.figure(figsize=(16.8, 6.6), facecolor="white")
    ax = fig.add_subplot(111)
    draw_sankey(ax, source_counts, task_counts, pair_counts, fontsize)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RefAmb source distribution as a pie chart or Sankey diagram.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the RefAmb jsonl dataset.")
    parser.add_argument("--figure", choices=["pie", "sankey"], default="sankey", help="Which figure to generate.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Output image path.")
    parser.add_argument("--fontsize", type=int, default=18, help="Base font size for labels inside the figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_counts, task_counts, pair_counts = load_counts(args.data)
    if not source_counts:
        raise ValueError(f"No data found in {args.data}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.figure == "pie":
        fig = build_pie_figure(source_counts, args.fontsize)
    else:
        fig = build_sankey_figure(source_counts, task_counts, pair_counts, args.fontsize)
    fig.savefig(args.out, format=args.out.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
