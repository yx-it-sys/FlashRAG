from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


PROJECT_ROOT = Path("/home/you/FlashRAG/exps/idea10")
PLOT_SCRIPT = PROJECT_ROOT / "scripts" / "plot_refamb_source_pie.py"
FIG_DIR = PROJECT_ROOT / "idea_reports/refamb_analysis_figs"
OUT_PDF = FIG_DIR / "refamb_original_balanced_source_pie_sankey_combined.pdf"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data/datasets/RefAmb/original_balanced.jsonl"
DEFAULT_FONTSIZE = 20


def load_plot_module():
    spec = importlib.util.spec_from_file_location("refamb_source_plot", PLOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load plotting module from {PLOT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the two RefAmb subfigures and combine them into one PDF.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the RefAmb jsonl dataset.")
    parser.add_argument("--fontsize", type=int, default=DEFAULT_FONTSIZE, help="Base font size used for both subfigures and the panel labels.")
    parser.add_argument("--out", type=Path, default=OUT_PDF, help="Output combined PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)

    module = load_plot_module()
    source_counts, task_counts, pair_counts = module.load_counts(args.data)
    if not source_counts:
        raise ValueError(f"No data found in {args.data}")

    module.apply_serif_style()

    # Start with an approximate canvas, draw once, then resize from the actual rendered bounds.
    fig = plt.figure(figsize=(26.0, 8.8), facecolor="white")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.9], wspace=0.16)
    ax_pie = fig.add_subplot(gs[0, 0])
    ax_sankey = fig.add_subplot(gs[0, 1])

    module.draw_pie(ax_pie, source_counts, args.fontsize)
    module.draw_sankey(ax_sankey, source_counts, task_counts, pair_counts, args.fontsize)

    # Titles are added after the first draw so their positions are included in the canvas fit.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    # Guard against any backend quirks by adding a small padding after measuring.
    pad_w = 0.35
    pad_h = 0.35
    fig.set_size_inches(bbox.width + pad_w, bbox.height + pad_h)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    pie_bbox = ax_pie.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    sankey_bbox = ax_sankey.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    pie_center = pie_bbox.x0 + pie_bbox.width / 2.0
    sankey_center = sankey_bbox.x0 + sankey_bbox.width / 2.0

    fig.text(pie_center, 0.02, "(a) RefAmb source distribution", ha="center", va="bottom", fontsize=args.fontsize, family="serif", weight="bold")
    fig.text(sankey_center, 0.02, "(b) Source-task Sankey diagram", ha="center", va="bottom", fontsize=args.fontsize, family="serif", weight="bold")

    fig.canvas.draw()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
