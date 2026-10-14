#!/usr/bin/env python3

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BASE_DIR = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B"
)
STATS_DIR = BASE_DIR / "stats"
LABEL_GLOB = "RefAmb_*_refamb_*_stage/label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.jsonl"

DATASET_ORDER = ["CRAG", "InfoSeek", "MCSearch", "OVEN"]
CATEGORY_ORDER = [
    "Object Identification",
    "Description",
    "Indirect Entity Ambiguity",
    "No Object Involved",
    "Mixed",
]
CATEGORY_COLORS = {
    "Object Identification": "#2F6C8F",
    "Description": "#E07A5F",
    "Indirect Entity Ambiguity": "#6A994E",
    "No Object Involved": "#7B6D8D",
    "Mixed": "#C9A227",
}
DATASET_NAME_MAP = {
    "crag": "CRAG",
    "infoseek": "InfoSeek",
    "mcsearch": "MCSearch",
    "oven": "OVEN",
}

JSON_OUTPUT_PATH = STATS_DIR / "entity_ambiguity_category_distribution.json"
PDF_OUTPUT_PATH = STATS_DIR / "entity_ambiguity_category_distribution.pdf"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_dataset_name(path: Path) -> str:
    stage_name = path.parts[-4]
    for key, dataset_name in DATASET_NAME_MAP.items():
        if key in stage_name.lower():
            return dataset_name
    raise ValueError(f"Unable to infer dataset name from: {path}")


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


def ratio_map(counts: dict[str, int], denominator: int) -> dict[str, float]:
    if denominator == 0:
        return {category: 0.0 for category in CATEGORY_ORDER}
    return {
        category: counts.get(category, 0) / denominator for category in CATEGORY_ORDER
    }


def build_distribution() -> dict:
    per_dataset_sample_categories: dict[str, dict[str, str]] = defaultdict(dict)

    for label_path in sorted(BASE_DIR.glob(LABEL_GLOB)):
        dataset_name = infer_dataset_name(label_path)
        for row in read_jsonl(label_path):
            per_dataset_sample_categories[dataset_name][row["id"]] = sample_level_category(row)

    ambiguous_counts_by_dataset = {}
    non_ambiguous_counts_by_dataset = {}
    for dataset_name in DATASET_ORDER:
        counter = Counter(per_dataset_sample_categories.get(dataset_name, {}).values())
        ambiguous_counts_by_dataset[dataset_name] = {
            category: int(counter.get(category, 0)) for category in CATEGORY_ORDER
        }
        non_ambiguous_counts_by_dataset[dataset_name] = int(counter.get("No", 0))

    overall_counter = Counter()
    total_non_ambiguous = 0
    for dataset_name in DATASET_ORDER:
        overall_counter.update(ambiguous_counts_by_dataset[dataset_name])
        total_non_ambiguous += non_ambiguous_counts_by_dataset[dataset_name]

    overall_ambiguous_total = sum(overall_counter.values())
    overall_total = overall_ambiguous_total + total_non_ambiguous

    result = {
        "base_dir": str(BASE_DIR),
        "category_rule": (
            "Sample-level aggregation over labeled search steps. "
            "If exactly one unique ambiguity type appears among steps with "
            "entity_ambiguous=Yes, assign that type. If multiple unique types "
            "appear, assign Mixed. If no step is ambiguous, assign No."
        ),
        "categories": CATEGORY_ORDER,
        "overall": {
            "sample_total": overall_total,
            "ambiguous_total": overall_ambiguous_total,
            "non_ambiguous_total": total_non_ambiguous,
            "counts": {category: int(overall_counter[category]) for category in CATEGORY_ORDER},
        },
        "by_dataset": {},
    }

    result["overall"]["ratios_within_ambiguous"] = ratio_map(
        result["overall"]["counts"], overall_ambiguous_total
    )
    result["overall"]["ratios_within_all_samples"] = ratio_map(
        result["overall"]["counts"], overall_total
    )

    for dataset_name in DATASET_ORDER:
        counts = ambiguous_counts_by_dataset[dataset_name]
        ambiguous_total = sum(counts.values())
        non_ambiguous_total = non_ambiguous_counts_by_dataset[dataset_name]
        sample_total = ambiguous_total + non_ambiguous_total
        result["by_dataset"][dataset_name] = {
            "sample_total": sample_total,
            "ambiguous_total": ambiguous_total,
            "non_ambiguous_total": non_ambiguous_total,
            "counts": counts,
            "ratios_within_ambiguous": ratio_map(counts, ambiguous_total),
            "ratios_within_all_samples": ratio_map(counts, sample_total),
        }

    return result


def write_json(distribution: dict) -> None:
    JSON_OUTPUT_PATH.write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def plot_distribution(distribution: dict) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
        constrained_layout=True,
    )

    overall = distribution["overall"]
    counts = [overall["counts"][category] for category in CATEGORY_ORDER]
    ratios = [overall["ratios_within_ambiguous"][category] for category in CATEGORY_ORDER]
    colors = [CATEGORY_COLORS[category] for category in CATEGORY_ORDER]

    ax0 = axes[0]
    bars = ax0.bar(range(len(CATEGORY_ORDER)), counts, color=colors, width=0.72)
    ax0.set_xticks(range(len(CATEGORY_ORDER)))
    ax0.set_xticklabels(
        [
            "Object\nIdentification",
            "Description",
            "Indirect Entity\nAmbiguity",
            "No Object\nInvolved",
            "Mixed",
        ]
    )
    ax0.set_ylabel("Ambiguous sample count")
    ax0.set_title("Overall 5-category distribution")
    ax0.grid(axis="y", linestyle="--", alpha=0.28)
    ax0.set_axisbelow(True)
    for bar, ratio in zip(bars, ratios):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ratio:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax1 = axes[1]
    x_positions = range(len(DATASET_ORDER))
    bottoms = [0.0] * len(DATASET_ORDER)
    for category in CATEGORY_ORDER:
        values = [
            distribution["by_dataset"][dataset_name]["ratios_within_ambiguous"][category]
            for dataset_name in DATASET_ORDER
        ]
        ax1.bar(
            x_positions,
            values,
            bottom=bottoms,
            label=category,
            color=CATEGORY_COLORS[category],
            width=0.68,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax1.set_xticks(list(x_positions))
    ax1.set_xticklabels(DATASET_ORDER)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Share within ambiguous samples")
    ax1.set_title("Per-dataset composition")
    ax1.grid(axis="y", linestyle="--", alpha=0.28)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=10, loc="upper right")

    for idx, dataset_name in enumerate(DATASET_ORDER):
        ambiguous_total = distribution["by_dataset"][dataset_name]["ambiguous_total"]
        ax1.text(
            idx,
            1.02,
            f"n={ambiguous_total}",
            ha="center",
            va="bottom",
            fontsize=10,
            transform=ax1.get_xaxis_transform(),
        )

    fig.savefig(PDF_OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    distribution = build_distribution()
    write_json(distribution)
    plot_distribution(distribution)

    print(f"Saved JSON to {JSON_OUTPUT_PATH}")
    print(f"Saved PDF to {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
