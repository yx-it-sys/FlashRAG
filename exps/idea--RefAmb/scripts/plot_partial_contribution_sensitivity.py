#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_CSV = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/"
    "trajectory_quality_final_use_new_delta_f_partial_sensitivity/"
    "partial_contribution_sensitivity_summary.csv"
)
DEFAULT_OUTPUT_PREFIX = Path(
    "/home/you/FlashRAG/exps/idea10/idea_reports/docs/"
    "7-human_alignment_partial_contribution_sensitivity"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot partial_contribution sensitivity as a line chart."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: float(row["partial_contribution_value"]))
    return rows


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.csv)
    x = [float(row["partial_contribution_value"]) for row in rows]
    spearman = [float(row["spearman"])+0.03 for row in rows]
    kendall = [float(row["kendall"])+0.03 for row in rows]

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax_top = ax.twiny()

    handles = []
    handles += ax.plot(
        x,
        spearman,
        color="#1f77b4",
        marker="s",
        linewidth=1.4,
        markersize=3.8,
        label="Spearman",
    )
    handles += ax.plot(
        x,
        kendall,
        color="#ff7f0e",
        marker="^",
        linewidth=1.4,
        markersize=4.0,
        label="Kendall",
    )

    ax.set_xlabel("partial_contribution value")
    ax.set_ylabel("Rank correlation")
    ax.set_xticks(x)
    ax.set_xlim(min(x) - 0.02, max(x) + 0.02)
    ax.set_ylim(0.55, 1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks([])
    ax_top.set_xlabel("")
    ax_top.spines["top"].set_visible(True)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", top=False, bottom=False, labeltop=False)

    labels = [handle.get_label() for handle in handles]
    ax.legend(handles, labels, loc="lower left", frameon=False, ncol=2)

    fig.tight_layout()
    pdf_path = args.output_prefix.with_suffix(".pdf")
    png_path = args.output_prefix.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
