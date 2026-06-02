#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "figs" / "interventions" / "trajectory_length_distribution"
DEFAULT_PREFIX = "trajectory_length_distribution_full_universe_multi_v1"

MODEL_SPECS = {
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
    "InternVL3.5-8B": {
        "result_root": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_InternVL3.5-8B"),
        "label": "InternVL3.5-8B",
    },
}

SETTING_PATHS = {
    "Original": "trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
    "Disturb": "first_round_disturb_rewrite_static_prefix_whole/trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
    "Oracle": "first_round_oracle_rewrite/trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
}

SETTING_STYLE = {
    "Original": {"color": "#406D96", "marker": "o"},
    "Disturb": {"color": "#C46E2E", "marker": "^"},
    "Oracle": {"color": "#2E8B57", "marker": "s"},
}

SETTING_ORDER = ["Original", "Disturb", "Oracle"]
LENGTHS = [0, 1, 2, 3, 4, 5]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_setting_samples(result_root: Path, stage_names: list[str], setting: str) -> list[dict]:
    rel_path = SETTING_PATHS[setting]
    samples: list[dict] = []
    for stage in stage_names:
        stage_dir = result_root / stage
        candidates = [
            stage_dir / rel_path,
            stage_dir / rel_path.replace("trajectory_quality_eval_whole_delta_F_updated", "trajectory_quality_eval_whole_delta_F"),
        ]
        chosen = next((path for path in candidates if path.exists()), None)
        if chosen is None:
            raise FileNotFoundError(candidates[0])
        samples.extend(list(load_jsonl(chosen)))
    return samples


def build_stage_full_universe_records(stage_dir: Path) -> dict[str, list[dict]]:
    original_rows = load_setting_samples(stage_dir.parent, [stage_dir.name], "Original")
    disturb_rows = load_setting_samples(stage_dir.parent, [stage_dir.name], "Disturb")
    oracle_rows = load_setting_samples(stage_dir.parent, [stage_dir.name], "Oracle")

    original_map = {row["id"]: row for row in original_rows if row.get("id")}
    disturb_map = {row["id"]: row for row in disturb_rows if row.get("id")}
    oracle_map = {row["id"]: row for row in oracle_rows if row.get("id")}

    records = {setting: [] for setting in SETTING_ORDER}
    for sample_id, original_row in original_map.items():
        records["Original"].append(original_row)
        records["Disturb"].append(disturb_map.get(sample_id, original_row))
        records["Oracle"].append(oracle_map.get(sample_id, original_row))
    return records


def summarize_model(result_root: Path) -> dict[str, list[dict]]:
    stage_names = sorted([p.name for p in result_root.iterdir() if p.is_dir() and p.name.endswith("_stage")])
    pooled = {setting: [] for setting in SETTING_ORDER}
    for stage_name in stage_names:
        stage_dir = result_root / stage_name
        stage_records = build_stage_full_universe_records(stage_dir)
        for setting in SETTING_ORDER:
            pooled[setting].extend(stage_records[setting])
    return pooled


def collect_length_distribution(records: list[dict]) -> dict:
    counts = {length: 0 for length in LENGTHS}
    total = len(records)
    for row in records:
        length = len(row.get("steps", []))
        if length not in counts:
            length = max(LENGTHS)
        counts[length] += 1

    probs = {length: (counts[length] / total if total else 0.0) for length in LENGTHS}
    return {
        "counts": [counts[length] for length in LENGTHS],
        "probs": [probs[length] for length in LENGTHS],
        "total_trajectories": total,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot trajectory length distributions for multiple models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SPECS.keys()),
        choices=list(MODEL_SPECS.keys()),
        help="Which model result directories to include.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for the figure and summary files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Output filename prefix.",
    )
    return parser


def plot_compact_grid(model_summaries: list[tuple[str, dict]], out_dir: Path, prefix: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "mathtext.fontset": "dejavuserif",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.titleweight": "bold",
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.pad_inches": 0.05,
        }
    )

    fig = plt.figure(figsize=(20, 10.5))
    gs = GridSpec(2, 6, figure=fig, wspace=0.22, hspace=0.30)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]

    bar_width = 0.24
    x = np.arange(len(LENGTHS))
    offsets = {
        "Original": -bar_width,
        "Disturb": 0.0,
        "Oracle": bar_width,
    }

    for ax, (model_label, summary) in zip(axes, model_summaries):
        for setting in SETTING_ORDER:
            style = SETTING_STYLE[setting]
            probs = summary[setting]["probs"]
            ax.bar(
                x + offsets[setting],
                probs,
                width=bar_width,
                color=style["color"],
                label=setting,
                alpha=0.96,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(model_label, fontsize=13, fontweight="bold", pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(length) for length in LENGTHS], fontsize=11)
        ax.set_ylim(0, 0.6)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

    fig.supylabel("Probability", fontsize=13, fontweight="bold", x=0.01)
    fig.supxlabel("Trajectory length", fontsize=13, fontweight="bold", y=0.04)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        framealpha=0.95,
        fancybox=True,
        bbox_to_anchor=(0.5, 0.985),
    )

    pdf_path = out_dir / f"{prefix}.pdf"
    png_path = out_dir / f"{prefix}.png"
    fig.subplots_adjust(left=0.05, right=0.995, top=0.88, bottom=0.10, wspace=0.18, hspace=0.35)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def write_summary(model_summaries: list[tuple[str, dict]], out_dir: Path, prefix: str) -> tuple[Path, Path]:
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "setting", "length", "count", "probability", "total_trajectories"])
        payload: dict[str, dict] = {}
        for model_label, summary in model_summaries:
            payload[model_label] = summary
            for setting in SETTING_ORDER:
                total = summary[setting]["total_trajectories"]
                for idx, length in enumerate(LENGTHS):
                    writer.writerow(
                        [
                            model_label,
                            setting,
                            length,
                            summary[setting]["counts"][idx],
                            summary[setting]["probs"][idx],
                            total,
                        ]
                    )
    payload["csv"] = str(csv_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir

    model_summaries: list[tuple[str, dict]] = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        pooled = summarize_model(spec["result_root"])
        summary = {setting: collect_length_distribution(pooled[setting]) for setting in SETTING_ORDER}
        model_summaries.append((spec["label"], summary))

    pdf_path, png_path = plot_compact_grid(model_summaries, out_dir, args.prefix)
    csv_path, json_path = write_summary(model_summaries, out_dir, args.prefix)
    print(f"Saved compact figure: {pdf_path}")
    print(f"Saved compact image:  {png_path}")
    print(f"Saved summary csv:    {csv_path}")
    print(f"Saved summary json:   {json_path}")


if __name__ == "__main__":
    main()
