#!/usr/bin/env python3

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "font.size": 13,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)

BASE_DIR = Path("/home/you/FlashRAG/exps/idea10/data/result/scaling_qwen3_5")
OUTPUT_DIR = Path("/home/you/FlashRAG/exps/idea10/idea_reports/scaling_qwen3_5_entity_ambiguity")
MODEL_SIZES = ["4b", "9b", "27b"]
SOURCES = ["mcsearch", "oven", "infoseek", "crag"]
LABEL_REL_PATH = Path("label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.jsonl")

SUMMARY_JSON = OUTPUT_DIR / "scaling_entity_ambiguity_summary.json"
SUMMARY_CSV = OUTPUT_DIR / "scaling_entity_ambiguity_summary.csv"
PER_SOURCE_CSV = OUTPUT_DIR / "scaling_entity_ambiguity_per_source.csv"
ITEM_RATIO_PDF = OUTPUT_DIR / "scaling_ambiguity_item_ratio.pdf"
ITEM_RATIO_PNG = OUTPUT_DIR / "scaling_ambiguity_item_ratio.png"
STEP_RATIO_PDF = OUTPUT_DIR / "scaling_ambiguity_step_ratio.pdf"
STEP_RATIO_PNG = OUTPUT_DIR / "scaling_ambiguity_step_ratio.png"

COLOR_OVERALL = "#C85C5C"
SOURCE_COLORS = {
    "mcsearch": "#3B82F6",
    "oven": "#F59E0B",
    "infoseek": "#10B981",
    "crag": "#8B5CF6",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_text_retrieval_search_step(step: dict) -> bool:
    if step.get("action") != "search":
        return False
    content = str(step.get("content") or "")
    return "Text Retrieval" in content


def is_entity_ambiguous(step: dict) -> bool:
    llm_label = step.get("llm_label") or {}
    return llm_label.get("entity_ambiguous") == "Yes"


def compute_stats(rows: list[dict]) -> dict[str, float | int]:
    valid_item_count = 0
    ambiguous_item_count = 0
    total_text_query_steps = 0
    ambiguous_text_query_steps = 0
    skipped_empty_trajectory_items = 0

    for row in rows:
        trajectory = row.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) == 0:
            skipped_empty_trajectory_items += 1
            continue

        valid_item_count += 1
        item_has_ambiguous_query = False

        for step in trajectory:
            if not isinstance(step, dict):
                continue
            if not is_text_retrieval_search_step(step):
                continue
            total_text_query_steps += 1
            if is_entity_ambiguous(step):
                ambiguous_text_query_steps += 1
                item_has_ambiguous_query = True

        if item_has_ambiguous_query:
            ambiguous_item_count += 1

    item_ratio = (
        ambiguous_item_count / valid_item_count if valid_item_count > 0 else 0.0
    )
    step_ratio = (
        ambiguous_text_query_steps / total_text_query_steps
        if total_text_query_steps > 0
        else 0.0
    )
    return {
        "valid_item_count": valid_item_count,
        "ambiguous_item_count": ambiguous_item_count,
        "item_ratio": item_ratio,
        "total_text_query_steps": total_text_query_steps,
        "ambiguous_text_query_steps": ambiguous_text_query_steps,
        "step_ratio": step_ratio,
        "skipped_empty_trajectory_items": skipped_empty_trajectory_items,
    }


def build_summary() -> tuple[list[dict], list[dict], dict]:
    overall_rows = []
    per_source_rows = []
    payload = {"overall": {}, "per_source": defaultdict(dict)}

    for model_size in MODEL_SIZES:
        size_rows = []
        for source in SOURCES:
            path = BASE_DIR / model_size / source / LABEL_REL_PATH
            rows = read_jsonl(path)
            stats = compute_stats(rows)
            per_source_row = {
                "model_size": model_size,
                "source": source,
                **stats,
            }
            per_source_rows.append(per_source_row)
            payload["per_source"][source][model_size] = stats

            size_rows.extend(rows)

        overall_stats = compute_stats(size_rows)
        overall_row = {"model_size": model_size, **overall_stats}
        overall_rows.append(overall_row)
        payload["overall"][model_size] = overall_stats

    payload["per_source"] = dict(payload["per_source"])
    return overall_rows, per_source_rows, payload


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_percent(value: float) -> float:
    return value * 100.0


def plot_metric(
    overall_rows: list[dict],
    per_source_rows: list[dict],
    metric_key: str,
    ylabel: str,
    pdf_path: Path,
    png_path: Path,
) -> None:
    x = list(range(len(MODEL_SIZES)))
    x_labels = MODEL_SIZES

    fig, ax = plt.subplots(figsize=(7.8, 5.0))

    overall_y = [format_percent(row[metric_key]) for row in overall_rows]
    ax.plot(
        x,
        overall_y,
        marker="o",
        linewidth=2.6,
        markersize=7,
        color=COLOR_OVERALL,
        label="Overall",
    )

    rows_by_source = defaultdict(list)
    for row in per_source_rows:
        rows_by_source[row["source"]].append(row)
    for source in SOURCES:
        source_rows = sorted(rows_by_source[source], key=lambda r: MODEL_SIZES.index(r["model_size"]))
        y = [format_percent(row[metric_key]) for row in source_rows]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            markersize=5.5,
            color=SOURCE_COLORS[source],
            alpha=0.9,
            label=source,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Model Size")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_rows, per_source_rows, payload = build_summary()

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    summary_fieldnames = [
        "model_size",
        "valid_item_count",
        "ambiguous_item_count",
        "item_ratio",
        "total_text_query_steps",
        "ambiguous_text_query_steps",
        "step_ratio",
        "skipped_empty_trajectory_items",
    ]
    per_source_fieldnames = [
        "model_size",
        "source",
        "valid_item_count",
        "ambiguous_item_count",
        "item_ratio",
        "total_text_query_steps",
        "ambiguous_text_query_steps",
        "step_ratio",
        "skipped_empty_trajectory_items",
    ]
    write_csv(SUMMARY_CSV, overall_rows, summary_fieldnames)
    write_csv(PER_SOURCE_CSV, per_source_rows, per_source_fieldnames)

    plot_metric(
        overall_rows=overall_rows,
        per_source_rows=per_source_rows,
        metric_key="item_ratio",
        ylabel="Ambiguous-item Ratio (%)",
        pdf_path=ITEM_RATIO_PDF,
        png_path=ITEM_RATIO_PNG,
    )
    plot_metric(
        overall_rows=overall_rows,
        per_source_rows=per_source_rows,
        metric_key="step_ratio",
        ylabel="Ambiguous-step Ratio (%)",
        pdf_path=STEP_RATIO_PDF,
        png_path=STEP_RATIO_PNG,
    )

    print(f"Saved summary JSON to {SUMMARY_JSON}")
    print(f"Saved summary CSV to {SUMMARY_CSV}")
    print(f"Saved per-source CSV to {PER_SOURCE_CSV}")
    print(f"Saved item-ratio plots to {ITEM_RATIO_PDF} and {ITEM_RATIO_PNG}")
    print(f"Saved step-ratio plots to {STEP_RATIO_PDF} and {STEP_RATIO_PNG}")
    print("Overall rows:")
    for row in overall_rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
