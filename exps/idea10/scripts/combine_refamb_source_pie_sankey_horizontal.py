from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path("/home/you/FlashRAG/exps/idea10")
PIE_SCRIPT = PROJECT_ROOT / "scripts" / "plot_refamb_source_pie.py"
SANKEY_SCRIPT = PROJECT_ROOT / "scripts" / "plot_refamb_source_task_sankey_balanced.py"
FIG_DIR = PROJECT_ROOT / "idea_reports/refamb_analysis_figs"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data/datasets/RefAmb/original_balanced.jsonl"
DEFAULT_OUT = FIG_DIR / "refamb_original_balanced_source_pie_sankey_combined_horizontal.pdf"


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine RefAmb source pie and Sankey into a horizontal two-panel PDF.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the RefAmb jsonl dataset.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output combined PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)

    pie_mod = load_module(PIE_SCRIPT, "refamb_source_pie")
    sankey_mod = load_module(SANKEY_SCRIPT, "refamb_source_task_sankey")
    source_counts, task_counts, pair_counts = pie_mod.load_counts(args.data)
    if not source_counts:
        raise ValueError(f"No data found in {args.data}")

    pie_mod.apply_serif_style()

    fig = plt.figure(figsize=(15.0, 6.3), facecolor="white")
    ax_pie = fig.add_axes([0.00, 0.14, 0.46, 0.80])
    ax_sankey = fig.add_axes([0.40, 0.14, 0.58, 0.80])

    pie_mod.draw_pie(ax_pie, source_counts, fontsize=17)

    ax_sankey.set_facecolor("white")
    left_x0, left_x1 = 0.06, 0.18
    right_x0, right_x1 = 0.80, 0.92
    top, bottom = 0.93, 0.08
    source_order = [k for k, _ in sankey_mod.SOURCE_LABELS]
    task_order = sankey_mod.TASK_LABELS
    source_pos, _ = sankey_mod.build_layout(source_counts, source_order, top=top, bottom=bottom, gap=0.022)
    task_pos, _ = sankey_mod.build_layout(task_counts, task_order, top=top, bottom=bottom, gap=0.022)
    source_offsets = {k: source_pos[k][0] for k in source_order}
    task_offsets = {k: task_pos[k][0] for k in task_order}

    for source_key, _ in sankey_mod.SOURCE_LABELS:
        for task in task_order:
            value = pair_counts[(source_key, task)]
            if value <= 0:
                continue
            sy0 = source_offsets[source_key]
            sy1 = sy0 + value * (source_pos[source_key][1] - source_pos[source_key][0]) / source_counts[source_key]
            ty0 = task_offsets[task]
            ty1 = ty0 + value * (task_pos[task][1] - task_pos[task][0]) / task_counts[task]
            sankey_mod.add_ribbon(
                ax_sankey,
                left_x1,
                right_x0,
                sy0,
                sy1,
                ty0,
                ty1,
                (sankey_mod.SOURCE_COLORS[source_key][0], sankey_mod.SOURCE_COLORS[source_key][1], sankey_mod.SOURCE_COLORS[source_key][2], 0.58),
            )
            source_offsets[source_key] = sy1
            task_offsets[task] = ty1

    node_width = left_x1 - left_x0
    right_node_width = right_x1 - right_x0
    for source_key, display_name in sankey_mod.SOURCE_LABELS:
        y0, y1 = source_pos[source_key]
        ax_sankey.add_patch(
            Rectangle((left_x0, y0), node_width, y1 - y0, facecolor=sankey_mod.SOURCE_COLORS[source_key], edgecolor="none", linewidth=0.0)
        )
        ax_sankey.text((left_x0 + left_x1) / 2, (y0 + y1) / 2, display_name, ha="center", va="center", fontsize=11.5, family="serif", weight="bold")

    for task in task_order:
        y0, y1 = task_pos[task]
        ax_sankey.add_patch(
            Rectangle((right_x0, y0), right_node_width, y1 - y0, facecolor=sankey_mod.TASK_COLORS[task], edgecolor="none", linewidth=0.0)
        )
        ax_sankey.text((right_x0 + right_x1) / 2, (y0 + y1) / 2, sankey_mod.task_label_display(task), ha="center", va="center", fontsize=12.2, family="serif", weight="bold")

    ax_sankey.text(0.12, 0.985, "Source", ha="center", va="top", fontsize=16, family="serif", weight="bold")
    ax_sankey.text(0.86, 0.985, "Task Type", ha="center", va="top", fontsize=16, family="serif", weight="bold")
    ax_sankey.set_xlim(0.0, 1.0)
    ax_sankey.set_ylim(0.0, 1.0)
    ax_sankey.axis("off")

    fig.text(0.23, 0.03, "(a) RefAmb source distribution", ha="center", va="bottom", fontsize=16, family="serif", weight="bold")
    fig.text(0.70, 0.03, "(b) Source-to-task Sankey Flow", ha="center", va="bottom", fontsize=16, family="serif", weight="bold")

    fig.savefig(args.out, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
