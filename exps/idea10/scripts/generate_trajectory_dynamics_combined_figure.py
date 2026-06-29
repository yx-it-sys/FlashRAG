#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np


ROOT = Path("/home/you/FlashRAG/exps/idea10")
TOP_IMAGE_CANDIDATES = [
    ROOT / "idea_reports/papers/figs/refamb_step_score_curve_ambiguity_split.png",
    ROOT / "idea_reports/papers/figs/refamb_step_score_curve_no_len_filter.png",
]
BOTTOM_IMAGE = ROOT / "data/result/RefAmb_original_Qwen2.5-vl-7B/stats/delta_f_trajectories_by_iteration.png"
OUT_DIR = ROOT / "research_analysis/main/refamb_latex_project/paper_figs"
OUT_PDF = OUT_DIR / "trajectory_dynamics_combined.pdf"
OUT_PNG = OUT_DIR / "trajectory_dynamics_combined.png"

TITLE_FONT = fm.FontProperties(family="Noto Sans CJK JP", weight="bold")


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return plt.imread(path)


def resolve_top_image() -> Path:
    for path in TOP_IMAGE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(TOP_IMAGE_CANDIDATES[0])


def add_panel(ax: plt.Axes, image_path: Path, title: str) -> None:
    image = load_image(image_path)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title, fontproperties=TITLE_FONT, fontsize=20, pad=12)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(20.5, 13.2),
        constrained_layout=True,
    )
    top_image = resolve_top_image()
    add_panel(
        axes[0],
        top_image,
        r"(a) Step-score decay by ambiguity status",
    )
    add_panel(
        axes[1],
        BOTTOM_IMAGE,
        r"(b) Factual novelty decay $\Delta F$",
    )

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(OUT_PDF)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
