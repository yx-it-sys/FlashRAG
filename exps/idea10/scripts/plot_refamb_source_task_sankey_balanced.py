from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


PROJECT_ROOT = Path("/home/you/FlashRAG/exps/idea10")
DATA_PATH = PROJECT_ROOT / "data/datasets/RefAmb/original_balanced.jsonl"
OUT_DIR = PROJECT_ROOT / "idea_reports/refamb_analysis_figs"
OUT_PDF = OUT_DIR / "refamb_original_balanced_source_task_sankey.pdf"
OUT_PNG = OUT_DIR / "refamb_original_balanced_source_task_sankey.png"


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
    "infoseek": (0x88 / 255, 0xB2 / 255, 0xDC / 255, 0.95),
    "mcsearch": (0x99 / 255, 0xD0 / 255, 0xC9 / 255, 0.95),
    "crag": (0xF0 / 255, 0xAE / 255, 0x67 / 255, 0.95),
    "oven": (0xBE / 255, 0x8C / 255, 0xB8 / 255, 0.95),
}

TASK_COLORS = {
    "Single-hop Attribute Query": (0xB9 / 255, 0xCF / 255, 0xE8 / 255, 1.0),
    "Entity Recognition": (0xCB / 255, 0xD5 / 255, 0xE6 / 255, 1.0),
    "Subproblem Aggregation": (0xC3 / 255, 0xDE / 255, 0xD8 / 255, 1.0),
    "Multi-hop": (0xEC / 255, 0xCA / 255, 0x9E / 255, 1.0),
    "Comparison": (0xDC / 255, 0xC8 / 255, 0xDE / 255, 1.0),
}

TEXT_COLOR = "black"
NODE_EDGE = (0.25, 0.25, 0.25, 0.85)


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
        return "S.A.Q."
    if task == "Entity Recognition":
        return "E.R."
    if task == "Subproblem Aggregation":
        return "C.C.Q."
    if task == "Multi-hop":
        return "M.H."
    if task == "Comparison":
        return "Comp."
    return task


def build_figure(source_counts: Counter, task_counts: Counter, pair_counts: Counter):
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    left_x0, left_x1 = 0.035, 0.15
    right_x0, right_x1 = 0.85, 0.965
    top, bottom = 0.93, 0.08

    source_order = [k for k, _ in SOURCE_LABELS]
    task_order = TASK_LABELS
    source_pos, _ = build_layout(source_counts, source_order, top=top, bottom=bottom, gap=0.022)
    task_pos, _ = build_layout(task_counts, task_order, top=top, bottom=bottom, gap=0.022)

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
            (SOURCE_COLORS[source_key][0], SOURCE_COLORS[source_key][1], SOURCE_COLORS[source_key][2], 0.58),
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
        rect = Rectangle(
            (left_x0, y0),
            node_width,
            y1 - y0,
            facecolor=SOURCE_COLORS[source_key],
            edgecolor="none",
            linewidth=0.0,
        )
        ax.add_patch(rect)
        ax.text(
            (left_x0 + left_x1) / 2,
            (y0 + y1) / 2,
            display_name,
            ha="center",
            va="center",
            fontsize=11.5,
            family="serif",
            weight="bold",
            color=TEXT_COLOR,
        )

    for task in task_order:
        y0, y1 = task_pos[task]
        rect = Rectangle(
            (right_x0, y0),
            right_node_width,
            y1 - y0,
            facecolor=TASK_COLORS[task],
            edgecolor="none",
            linewidth=0.0,
        )
        ax.add_patch(rect)
        ax.text(
            (right_x0 + right_x1) / 2,
            (y0 + y1) / 2,
            task_label_display(task),
            ha="center",
            va="center",
            fontsize=12.2,
            family="serif",
            weight="bold",
            color=TEXT_COLOR,
        )

    ax.text(0.03, 0.985, "Source", ha="left", va="top", fontsize=16, family="serif", weight="bold")
    ax.text(0.97, 0.985, "Task Type", ha="right", va="top", fontsize=16, family="serif", weight="bold")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.02)
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_counts, task_counts, pair_counts = load_counts(DATA_PATH)
    fig = build_figure(source_counts, task_counts, pair_counts)
    fig.text(0.5, 0.005, "(b) Source-to-task Sankey Flow", ha="center", va="bottom", fontsize=13.0, family="serif", weight="bold")
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT_PNG, format="png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
