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


BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 17
STEPS = [1, 2, 3, 4, 5]

MODEL_SPECS = {
    "Qwen2.5-vl-7B": {
        "result_root": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B"),
        "stages": [
            "RefAmb_2026_05_01_11_26_refamb_mcsearch_stage",
            "RefAmb_2026_05_01_13_28_refamb_oven_stage",
            "RefAmb_2026_05_02_13_50_refamb_infoseek_stage",
            "RefAmb_2026_05_02_14_22_refamb_crag_stage",
        ],
        "label": "Qwen2.5-vl-7B",
    },
    "Qwen3-vl-4B": {
        "result_root": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen3-vl-4B"),
        "stages": [
            "RefAmb_2026_05_23_11_40_refamb_oven_qwen3_vl_4b_stage",
            "RefAmb_2026_05_23_12_10_refamb_infoseek_qwen3_vl_4b_stage",
            "RefAmb_2026_05_23_12_53_refamb_mcsearch_qwen3_vl_4b_stage",
            "RefAmb_2026_05_23_14_07_refamb_crag_qwen3_vl_4b_stage",
        ],
        "label": "Qwen3-vl-4B",
    },
    "Qwen3-vl-8b": {
        "result_root": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen3-vl-8b"),
        "stages": [
            "RefAmb_2026_05_23_11_40_refamb_oven_qwen3_vl_8b_stage",
            "RefAmb_2026_05_23_12_10_refamb_infoseek_qwen3_vl_8b_stage",
            "RefAmb_2026_05_23_12_53_refamb_mcsearch_qwen3_vl_8b_stage",
            "RefAmb_2026_05_23_14_07_refamb_crag_qwen3_vl_8b_stage",
        ],
        "label": "Qwen3-vl-8b",
    },
    "Qwen3-vl-32B": {
        "result_root": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_qwen3_vl_32b"),
        "stages": [
            "RefAmb_2026_05_19_21_03_refamb_mcsearch_qwen3_vl_32b_stage",
            "RefAmb_2026_05_19_12_24_refamb_oven_qwen3_vl_32b_stage",
            "RefAmb_2026_05_19_15_53_refamb_infoseek_qwen3_vl_32b_stage",
            "RefAmb_2026_05_20_02_27_refamb_crag_qwen3_vl_32b_stage",
        ],
        "label": "Qwen3-vl-32B",
    },
    "InternVL3.5-8B": {
        "result_root": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_InternVL3.5-8B"),
        "stages": [
            "RefAmb_2026_05_09_15_53_refamb_oven_intervl3_5_8b_stage",
            "RefAmb_2026_05_09_16_13_refamb_infoseek_intervl3_5_8b_stage",
            "RefAmb_2026_05_09_16_39_refamb_mcsearch_intervl3_5_8b_stage",
            "RefAmb_2026_05_09_17_28_refamb_crag_intervl3_5_8b_stage",
        ],
        "label": "InternVL3.5-8B",
    },
}

SETTING_PATHS = {
    "Original": "trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
    "Oracle": "first_round_oracle_rewrite/trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
    "Disturb": "first_round_disturb_rewrite_static_prefix_whole/trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
}

SETTING_STYLE = {
    "Original": {"color": "#406D96", "shadow": "#A9C7E6", "marker": "o"},
    "Oracle": {"color": "#2E8B57", "shadow": "#A7D8B8", "marker": "s"},
    "Disturb": {"color": "#C46E2E", "shadow": "#F0C8A2", "marker": "^"},
}

SETTING_ORDER = ["Original", "Oracle", "Disturb"]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def bootstrap_mean_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean_value = float(arr.mean())
    if arr.size == 1:
        return mean_value, mean_value, mean_value
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(BOOTSTRAP_SAMPLES, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return (
        mean_value,
        float(np.quantile(boot_means, 0.025)),
        float(np.quantile(boot_means, 0.975)),
    )


def collect_step_score_means(samples: list[dict]) -> tuple[list[float], list[float], list[float], list[int], int]:
    buckets = {i: [] for i in STEPS}
    total = 0
    for row in samples:
        total += 1
        steps = row.get("steps", [])
        for i in STEPS:
            if len(steps) < i:
                continue
            val = steps[i - 1].get("step_score")
            if val is not None:
                buckets[i].append(float(val))

    means: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []
    counts_per_round: list[int] = []
    for i in STEPS:
        m, lo, hi = bootstrap_mean_ci(buckets[i])
        means.append(m)
        ci_low.append(lo)
        ci_high.append(hi)
        counts_per_round.append(len(buckets[i]))
    return means, ci_low, ci_high, counts_per_round, total


def load_setting_samples(result_root: Path, stage_names: list[str], setting: str) -> list[dict]:
    samples: list[dict] = []
    rel_path = SETTING_PATHS[setting]
    for stage in stage_names:
        stage_dir = result_root / stage
        candidates = [
            stage_dir / rel_path,
            stage_dir / rel_path.replace("trajectory_quality_eval_whole_delta_F_updated", "trajectory_quality_eval_whole_delta_F"),
        ]
        chosen = next((path for path in candidates if path.exists()), None)
        if chosen is None:
            raise FileNotFoundError(candidates[0])
        samples.extend(load_jsonl(chosen))
    return samples


def build_stage_full_universe_records(stage_dir: Path) -> list[dict]:
    original_rows = load_setting_samples(stage_dir.parent, [stage_dir.name], "Original")
    disturb_rows = load_setting_samples(stage_dir.parent, [stage_dir.name], "Disturb")
    oracle_rows = load_setting_samples(stage_dir.parent, [stage_dir.name], "Oracle")

    original_map = {row["id"]: row for row in original_rows if row.get("id")}
    disturb_map = {row["id"]: row for row in disturb_rows if row.get("id")}
    oracle_map = {row["id"]: row for row in oracle_rows if row.get("id")}

    records: list[dict] = []
    for sample_id, original_row in original_map.items():
        for setting, row_map in [
            ("Original", original_map),
            ("Disturb", disturb_map),
            ("Oracle", oracle_map),
        ]:
            row = row_map.get(sample_id, original_row)
            records.append(
                {
                    "id": sample_id,
                    "setting": setting,
                    "steps": row.get("steps", []),
                }
            )
    return records


def summarize_model(result_root: Path) -> dict:
    stage_names = sorted([p.name for p in result_root.iterdir() if p.is_dir() and p.name.endswith("_stage")])
    means_by_setting: dict[str, list[float]] = {}
    ci_low_by_setting: dict[str, list[float]] = {}
    ci_high_by_setting: dict[str, list[float]] = {}
    counts: dict[str, dict[str, object]] = {}

    pooled_records: dict[str, list[dict]] = {setting: [] for setting in SETTING_ORDER}
    for stage_name in stage_names:
        stage_dir = result_root / stage_name
        for record in build_stage_full_universe_records(stage_dir):
            pooled_records[record["setting"]].append(record)

    for setting in SETTING_ORDER:
        means, ci_low, ci_high, counts_per_round, total = collect_step_score_means(pooled_records[setting])
        means_by_setting[setting] = means
        ci_low_by_setting[setting] = ci_low
        ci_high_by_setting[setting] = ci_high
        counts[setting] = {
            "total_items": total,
            "n_items_with_this_round": counts_per_round,
        }

    return {
        "means_by_setting": means_by_setting,
        "ci_low_by_setting": ci_low_by_setting,
        "ci_high_by_setting": ci_high_by_setting,
        "counts": counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot RefAmb step score curves for one or more models.")
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
        default=Path("/home/you/FlashRAG/exps/idea10/idea_reports/new/refamb_latex_project/figs/interventions"),
        help="Output directory for the figure and summary files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="refamb_step_score_curve_no_len_filter_multi",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--annotate-counts",
        action="store_true",
        help="Annotate the curve with per-round sample counts in the exported table.",
    )
    return parser


def plot_multi_model_curves(model_summaries: list[tuple[str, dict]], out_dir: Path, prefix: str) -> tuple[Path, Path]:
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
            "axes.edgecolor": "black",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    n_rows = len(model_summaries)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, max(3.8 * n_rows, 4.6)), sharex=True)
    if n_rows == 1:
        axes = [axes]

    x = np.array(STEPS, dtype=float)
    for ax, (model_label, summary) in zip(axes, model_summaries):
        for setting in SETTING_ORDER:
            y = np.array(summary["means_by_setting"][setting], dtype=float)
            lo = np.array(summary["ci_low_by_setting"][setting], dtype=float)
            hi = np.array(summary["ci_high_by_setting"][setting], dtype=float)
            style = SETTING_STYLE[setting]
            ax.fill_between(x, lo, hi, color=style["shadow"], alpha=0.30, linewidth=0)
            ax.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                markersize=8,
                linewidth=2.6,
                label=setting,
            )

        ax.set_ylabel(model_label, fontsize=16, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ymax = max(max(summary["means_by_setting"][s]) for s in SETTING_ORDER)
        ax.set_ylim(0, ymax * 1.15 + 0.02)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        for label in ax.get_yticklabels():
            label.set_fontsize(12)
            label.set_fontweight("bold")

    axes[-1].set_xlabel("Iteration Round", fontsize=22, fontweight="bold")
    axes[-1].set_xticks(STEPS)
    axes[-1].set_xlim(0.85, 5.15)
    for label in axes[-1].get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight("bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{prefix}.pdf"
    png_path = out_dir / f"{prefix}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path


def write_summary_files(out_dir: Path, prefix: str, model_summaries: list[tuple[str, dict]]) -> tuple[Path, Path]:
    csv_path = out_dir / f"{prefix}_means.csv"
    json_path = out_dir / f"{prefix}_summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "setting", "round", "mean_step_score", "ci_low", "ci_high", "n_items_with_this_round"])
        for model_label, summary in model_summaries:
            for setting in SETTING_ORDER:
                for idx, round_id in enumerate(STEPS):
                    writer.writerow(
                        [
                            model_label,
                            setting,
                            round_id,
                            f"{summary['means_by_setting'][setting][idx]:.6f}",
                            f"{summary['ci_low_by_setting'][setting][idx]:.6f}",
                            f"{summary['ci_high_by_setting'][setting][idx]:.6f}",
                            summary['counts'][setting]['n_items_with_this_round'][idx],
                        ]
                    )

    payload = {
        model_label: summary
        for model_label, summary in model_summaries
    }
    payload["figure_pdf"] = str(out_dir / f"{prefix}.pdf")
    payload["figure_png"] = str(out_dir / f"{prefix}.png")
    payload["csv"] = str(csv_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir

    model_summaries: list[tuple[str, dict]] = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        summary = summarize_model(spec["result_root"])
        model_summaries.append((spec["label"], summary))

    pdf_path, png_path = plot_multi_model_curves(model_summaries, out_dir, args.prefix)
    csv_path, json_path = write_summary_files(out_dir, args.prefix, model_summaries)
    print("\n".join([str(pdf_path), str(png_path), str(csv_path), str(json_path)]))


if __name__ == "__main__":
    main()
