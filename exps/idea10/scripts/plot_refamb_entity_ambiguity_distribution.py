#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "font.size": 14,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BASE_DIR = Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B")
OUTPUT_DIR = Path("/home/you/FlashRAG/exps/idea10/idea_reports/refamb_analysis_figs")
OUTPUT_PDF = OUTPUT_DIR / "refamb_entity_ambiguity_category_distribution.pdf"

LABEL_GLOB = "RefAmb_*_refamb_*_stage/label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.jsonl"

DISPLAY_CATEGORY_ORDER = [
    "Object Identification",
    "Indirect Entity Ambiguity",
    "Description",
    "Mixed",
]

ALL_CATEGORY_ORDER = [
    "Object Identification",
    "Description",
    "Indirect Entity Ambiguity",
    "Mixed",
    "No Object Involved",
]

COLOR_MAP = {
    "Object Identification": "#6EA6D7",
    "Indirect Entity Ambiguity": "#F6C177",
    "Description": "#F2C0C0",
    "Mixed": "#8E6BB1",
    "No Object Involved": "#B9D4EE",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_level_category(record: dict) -> str:
    categories = []
    for step in record.get("trajectory", []):
        if step.get("action") != "search":
            continue
        llm_label = step.get("llm_label") or {}
        if llm_label.get("entity_ambiguous") != "Yes":
            continue
        category = llm_label.get("ambiguity_level")
        if category:
            categories.append(category)

    unique_categories = list(dict.fromkeys(categories))
    if not unique_categories:
        return "No"
    if len(unique_categories) == 1:
        return unique_categories[0]
    return "Mixed"


def build_distribution() -> tuple[dict[str, int], int]:
    counts = Counter()
    for label_path in sorted(BASE_DIR.glob(LABEL_GLOB)):
        for row in read_jsonl(label_path):
            counts[sample_level_category(row)] += 1

    ambiguous_total = sum(counts.get(category, 0) for category in ALL_CATEGORY_ORDER)
    return {category: int(counts.get(category, 0)) for category in ["No", *ALL_CATEGORY_ORDER]}, ambiguous_total


def plot_distribution(counts: dict[str, int], ambiguous_total: int) -> None:
    ratios = [
        (counts[category] / ambiguous_total * 100.0) if ambiguous_total else 0.0
        for category in DISPLAY_CATEGORY_ORDER
    ]

    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    y_positions = list(range(len(DISPLAY_CATEGORY_ORDER)))
    colors = [COLOR_MAP[category] for category in DISPLAY_CATEGORY_ORDER]

    ax.barh(y_positions, ratios, color=colors, height=0.56)
    ax.invert_yaxis()

    ax.set_xlim(0, max(70.0, max(ratios, default=0.0) * 1.18))
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_xticklabels([f"{tick}%" for tick in [0, 10, 20, 30, 40, 50, 60, 70]], fontsize=14)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(DISPLAY_CATEGORY_ORDER, rotation=28, ha="right", rotation_mode="anchor", fontsize=14)
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)

    for idx, ratio in enumerate(ratios):
        ax.text(
            ratio + 0.8,
            idx,
            f"{ratio:.1f}%",
            va="center",
            ha="left",
            fontsize=14,
            color="#333333",
        )

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.4)
        ax.spines[spine].set_color("#333333")

    ax.tick_params(axis="both", colors="#222222", length=0, labelsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(14)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.98))
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts, ambiguous_total = build_distribution()
    plot_distribution(counts, ambiguous_total)
    print(f"Saved PDF to {OUTPUT_PDF}")
    print(f"Counts: {counts}")
    print(f"Ambiguous total: {ambiguous_total}")


if __name__ == "__main__":
    main()
