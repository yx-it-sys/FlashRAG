from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


PROJECT_ROOT = Path("/home/you/FlashRAG/exps/idea10")
DATA_PATH = PROJECT_ROOT / "data/datasets/RefAmb/origin.jsonl"
OUT_DIR = PROJECT_ROOT / "idea_reports/refamb_analysis_figs"
OUT_PDF = OUT_DIR / "refamb_origin_source_task_sankey.pdf"
OUT_PNG = OUT_DIR / "refamb_origin_source_task_sankey.png"


SOURCE_LABELS = [
    ("infoseek", "InfoSeek"),
    ("mcsearch", "MC-Search"),
    ("crag", "CRAG-MM"),
    ("oven", "OVEN"),
]

TASK_LABELS = [
    "Single-hop Attribute Query",
    "Entity Recognition",
    "Subproblem Aggregation",
    "Multi-hop",
    "Comparison",
]

SOURCE_COLORS = {
    "infoseek": (31 / 255, 119 / 255, 180 / 255, 0.74),
    "mcsearch": (255 / 255, 127 / 255, 14 / 255, 0.74),
    "crag": (44 / 255, 160 / 255, 44 / 255, 0.74),
    "oven": (214 / 255, 39 / 255, 40 / 255, 0.74),
}

NODE_EDGE = (0.55, 0.55, 0.55, 0.9)
RIGHT_NODE_COLOR = (0.82, 0.82, 0.82, 0.65)
TEXT_COLOR = "black"


def load_counts(path: Path):
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


def build_layout(counts: Counter, order: list[str], top: float, bottom: float, gap: float = 0.02):
    total = sum(counts[k] for k in order)
    available = top - bottom - gap * (len(order) - 1)
    scale = available / total
    positions = {}
    cursor = top
    for key in order:
        h = counts[key] * scale
        positions[key] = (cursor - h, cursor)
        cursor = cursor - h - gap
    return positions, scale


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
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none")
    ax.add_patch(patch)


def task_label_display(task: str) -> str:
    if task == "Single-hop Attribute Query":
        return "Single-hop\nAttribute Query"
    if task == "Subproblem Aggregation":
        return "Subproblem\nAggregation"
    return task.replace(" ", "\n")


def build_figure(source_counts: Counter, task_counts: Counter, pair_counts: Counter):
    fig, ax = plt.subplots(figsize=(12.4, 5.7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    left_x0, left_x1 = 0.14, 0.24
    right_x0, right_x1 = 0.76, 0.86
    top, bottom = 0.90, 0.12

    source_order = [k for k, _ in SOURCE_LABELS]
    task_order = TASK_LABELS
    source_pos, _ = build_layout(source_counts, source_order, top=top, bottom=bottom, gap=0.028)
    task_pos, _ = build_layout(task_counts, task_order, top=top, bottom=bottom, gap=0.028)

    source_offsets = {k: source_pos[k][0] for k in source_order}
    task_offsets = {k: task_pos[k][0] for k in task_order}

    # Draw ribbons first so nodes sit on top.
    for source_key, _ in SOURCE_LABELS:
        links = [(task, pair_counts[(source_key, task)]) for task in task_order if pair_counts[(source_key, task)] > 0]
        for task, value in links:
            sy0 = source_offsets[source_key]
            sy1 = sy0 + value * (source_pos[source_key][1] - source_pos[source_key][0]) / source_counts[source_key]
            ty0 = task_offsets[task]
            ty1 = ty0 + value * (task_pos[task][1] - task_pos[task][0]) / task_counts[task]
            add_ribbon(
                ax,
                left_x1,
                right_x0,
                sy0,
                sy1,
                ty0,
                ty1,
                SOURCE_COLORS[source_key],
            )
            source_offsets[source_key] = sy1
            task_offsets[task] = ty1

    # Reset offsets for node drawing labels.
    source_offsets = {k: source_pos[k][0] for k in source_order}
    task_offsets = {k: task_pos[k][0] for k in task_order}

    # Draw nodes.
    node_width = left_x1 - left_x0
    right_node_width = right_x1 - right_x0
    for source_key, display_name in SOURCE_LABELS:
        y0, y1 = source_pos[source_key]
        rect = Rectangle((left_x0, y0), node_width, y1 - y0, facecolor=SOURCE_COLORS[source_key], edgecolor=NODE_EDGE, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(left_x0 - 0.03, (y0 + y1) / 2, f"{display_name}\n{source_counts[source_key]}", ha="right", va="center", fontsize=13, family="serif", color=TEXT_COLOR)

    for task in task_order:
        y0, y1 = task_pos[task]
        rect = Rectangle((right_x0, y0), right_node_width, y1 - y0, facecolor=RIGHT_NODE_COLOR, edgecolor=NODE_EDGE, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(right_x1 + 0.03, (y0 + y1) / 2, f"{task_label_display(task)}\n{task_counts[task]}", ha="left", va="center", fontsize=13, family="serif", color=TEXT_COLOR)

    ax.text(0.03, 0.97, "Source", ha="left", va="top", fontsize=16, family="serif", weight="bold")
    ax.text(0.97, 0.97, "Task Type", ha="right", va="top", fontsize=16, family="serif", weight="bold")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_counts, task_counts, pair_counts = load_counts(DATA_PATH)
    fig = build_figure(source_counts, task_counts, pair_counts)
    fig.savefig(OUT_PDF, format="pdf")
    fig.savefig(OUT_PNG, format="png", dpi=300)
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
