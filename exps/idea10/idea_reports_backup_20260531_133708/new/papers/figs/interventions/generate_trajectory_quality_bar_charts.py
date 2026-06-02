#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


# TASK_ORDER = [
#     "Entity Recognition",
#     "Single-hop Attribute Query",
#     "Multi-hop",
#     "Comparison",
#     "Subproblem Aggregation",
# ]

TASK_ORDER = [
    "Entity Recognition",
    "Single-hop Attribute Query",
    "Multi-hop",
    "Comparison",
    "Subproblem Aggregation",
]

TASK_DISPLAY_LABELS = {
    "Entity Recognition": "E.R.",
    "Single-hop Attribute Query": "S.A.Q.",
    "Multi-hop": "M.H.",
    "Comparison": "Comp.",
    "Subproblem Aggregation": "C.C.Q.",
}

TASK_DISPLAY_LABELS_WITH_AVG = [TASK_DISPLAY_LABELS[t] for t in TASK_ORDER] + ["Avg."]

SETTINGS = ["Original", "Disturb", "Oracle"]
SETTING_STYLE = {
    "Original": {"color": "#2B7DBA"},
    "Disturb": {"color": "#F28C28"},
    "Oracle": {"color": "#6DBB4C"},
}

DATASET_SPECS = {
    "InternVL3.5-8B": {
        "result_root": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_InternVL3.5-8B"),
        "label": "InternVL3.5-8B",
    },
    "Qwen2.5-vl-7B": {
        "result_root": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B"),
        "label": "Qwen2.5-vl-7B",
    },
    "Qwen3-vl-4B": {
        "result_root": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen3-vl-4B"),
        "label": "Qwen3-vl-4B",
    },
    "Qwen3-vl-8B": {
        "result_root": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen3-vl-8b"),
        "label": "Qwen3-vl-8B",
    },
    "Qwen3-vl-32B": {
        "result_root": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_qwen3_vl_32b"),
        "label": "Qwen3-vl-32B",
    },
}

SETTING_SUBDIR = {
    "Disturb": "first_round_disturb_rewrite_static_prefix_whole",
    "Oracle": "first_round_oracle_rewrite",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_tqs_path(stage_dir: Path, setting: str) -> Path:
    if setting == "Original":
        candidates = [
            stage_dir / "trajectory_quality_eval_whole_delta_F_updated" / "trajectory_quality_samples.jsonl",
            stage_dir / "trajectory_quality_eval_whole_delta_F" / "trajectory_quality_samples.jsonl",
        ]
    else:
        subdir = SETTING_SUBDIR[setting]
        candidates = [
            stage_dir / subdir / "trajectory_quality_eval_whole_delta_F_updated" / "trajectory_quality_samples.jsonl",
            stage_dir / subdir / "trajectory_quality_eval_whole_delta_F" / "trajectory_quality_samples.jsonl",
        ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Missing trajectory_quality_samples.jsonl for {setting} under {stage_dir}")


def metric_from_lj(row: dict) -> float:
    metric_score = row.get("output", {}).get("metric_score", {})
    for key in ["llm", "gpt_acc"]:
        val = metric_score.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    raise ValueError("No usable LJ field found")


def extract_rte(row: dict) -> float:
    return float(row.get("trajectory_quality_score", 0.0) or 0.0)


def mean_or_blank(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return None if not filtered else mean(filtered)


def build_stage_records(stage_dir: Path) -> list[dict]:
    original_metrics = load_json(stage_dir / "intermediate_data.json")
    original_metric_map = {row["id"]: row for row in original_metrics if row.get("id")}

    disturb_metrics = load_json(stage_dir / SETTING_SUBDIR["Disturb"] / "intermediate_data.json")
    disturb_metric_map = {row["id"]: row for row in disturb_metrics if row.get("id")}

    oracle_metrics = load_json(stage_dir / SETTING_SUBDIR["Oracle"] / "intermediate_data.json")
    oracle_metric_map = {row["id"]: row for row in oracle_metrics if row.get("id")}

    original_tqs_map = {row["id"]: row for row in load_jsonl(resolve_tqs_path(stage_dir, "Original")) if row.get("id")}
    disturb_tqs_map = {row["id"]: row for row in load_jsonl(resolve_tqs_path(stage_dir, "Disturb")) if row.get("id")}
    oracle_tqs_map = {row["id"]: row for row in load_jsonl(resolve_tqs_path(stage_dir, "Oracle")) if row.get("id")}

    records: list[dict] = []
    for sample_id, original_row in original_metric_map.items():
        task_type = original_row.get("task_type", "")
        if task_type not in TASK_ORDER:
            continue

        for setting, metric_map, tqs_map in [
            ("Original", original_metric_map, original_tqs_map),
            ("Disturb", disturb_metric_map, disturb_tqs_map),
            ("Oracle", oracle_metric_map, oracle_tqs_map),
        ]:
            metric_row = metric_map.get(sample_id)
            tqs_row = tqs_map.get(sample_id)
            if metric_row is None or tqs_row is None:
                continue
            records.append(
                {
                    "task_type": task_type,
                    "setting": setting,
                    "rte": extract_rte(tqs_row),
                    "lj_score": metric_from_lj(metric_row),
                }
            )
    return records


def build_dataset_summary(result_root: Path) -> dict:
    stage_dirs = sorted([p for p in result_root.iterdir() if p.is_dir() and p.name.endswith("_stage")])
    records: list[dict] = []
    for stage_dir in stage_dirs:
        records.extend(build_stage_records(stage_dir))

    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task_type"], record["setting"])].append(record)

    summary: dict[tuple[str, str], dict] = {}
    for task in TASK_ORDER:
        for setting in SETTINGS:
            items = grouped.get((task, setting), [])
            summary[(task, setting)] = {
                "n": len(items),
                "rte": mean_or_blank([it["rte"] for it in items]),
                "lj": mean_or_blank([it["lj_score"] for it in items]),
            }
    return {
        "records": records,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw grouped bar charts for RefAmb task-level RTE/LJ results.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_SPECS.keys()),
        choices=list(DATASET_SPECS.keys()),
        help="Datasets to include in the figure.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/you/FlashRAG/exps/idea10/idea_reports/new/papers/figs/interventions/bar_charts"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="refamb_trajectory_quality_bars",
        help="Filename prefix for outputs.",
    )
    return parser


def plot_grouped_bars(dataset_summaries: list[tuple[str, dict]], out_dir: Path, prefix: str) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.10,
        }
    )

    n_cols = len(dataset_summaries)
    metric_specs = [("rte", "RTE"), ("lj", "LJ Score")]
    fig, axes = plt.subplots(
        len(metric_specs),
        n_cols,
        figsize=(max(3.2 * n_cols, 7.6), 5.8),
        sharey="row",
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = [[axes[0]], [axes[1]]]

    width = 0.24
    offsets = {
        "Original": -width,
        "Disturb": 0.0,
        "Oracle": width,
    }

    for row_idx, (metric_key, metric_title) in enumerate(metric_specs):
        row_axes = axes[row_idx]
        for col_idx, (dataset_label, summary) in enumerate(dataset_summaries):
            ax = row_axes[col_idx]
            for setting in SETTINGS:
                means = []
                for task in TASK_ORDER:
                    cell = summary[(task, setting)]
                    means.append(cell[metric_key])
                valid_means = [value for value in means if value is not None]
                means.append(mean(valid_means) if valid_means else None)
                ax.bar(
                    [i + offsets[setting] for i in range(len(TASK_ORDER) + 1)],
                    [0.0 if v is None else v for v in means],
                    width=width,
                    color=SETTING_STYLE[setting]["color"],
                    label=setting if row_idx == 0 and col_idx == 0 else None,
                    edgecolor="white",
                    linewidth=0.6,
                )

            y_values = [summary[(task, setting)][metric_key] for task in TASK_ORDER for setting in SETTINGS if summary[(task, setting)][metric_key] is not None]
            if y_values:
                ymax = max(y_values)
                ax.set_ylim(0, ymax * 1.22 + 0.03)
            ax.set_xticks(list(range(len(TASK_ORDER) + 1)))
            ax.set_xticklabels(TASK_DISPLAY_LABELS_WITH_AVG, rotation=18, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            if row_idx == 0:
                ax.set_title(dataset_label)
            if col_idx == 0:
                ax.set_ylabel(metric_title)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(dataset_label.replace(" ", "_") for dataset_label, _ in dataset_summaries)
    pdf_path = out_dir / f"{prefix}_{tag}.pdf"
    png_path = out_dir / f"{prefix}_{tag}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    args = build_parser().parse_args()

    dataset_summaries: list[tuple[str, dict]] = []
    for dataset_key in args.datasets:
        spec = DATASET_SPECS[dataset_key]
        dataset_summaries.append((spec["label"], build_dataset_summary(spec["result_root"])["summary"]))

    outputs = plot_grouped_bars(dataset_summaries, args.out_dir, args.prefix)
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
