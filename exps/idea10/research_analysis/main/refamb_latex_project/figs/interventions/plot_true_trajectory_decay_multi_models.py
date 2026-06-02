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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
        "label": "Qwen3-vl-8B",
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
        "ci_lower_q": 0.10,
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


def bootstrap_mean_ci(vectors: np.ndarray, lower_q: float = 0.025) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if vectors.size == 0:
        zeros = np.zeros(len(STEPS), dtype=float)
        return zeros, zeros, zeros
    mean_value = vectors.mean(axis=0)
    if vectors.shape[0] == 1:
        return mean_value, mean_value, mean_value
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, vectors.shape[0], size=(BOOTSTRAP_SAMPLES, vectors.shape[0]))
    boot_means = vectors[idx].mean(axis=1)
    return (
        mean_value,
        np.quantile(boot_means, lower_q, axis=0),
        np.quantile(boot_means, 1.0 - lower_q, axis=0),
    )


def collect_trajectory_level_curve(samples: list[dict], lower_q: float = 0.025) -> dict:
    vectors: list[list[float]] = []
    alive_counts = [0 for _ in STEPS]
    total = 0

    for row in samples:
        total += 1
        steps = row.get("steps", [])
        vec: list[float] = []
        for idx in STEPS:
            if len(steps) >= idx:
                val = steps[idx - 1].get("step_score")
                if val is None:
                    val = 0.0
                else:
                    alive_counts[idx - 1] += 1
                    val = float(val)
            else:
                val = 0.0
            vec.append(val)
        vectors.append(vec)

    arr = np.asarray(vectors, dtype=float) if vectors else np.zeros((0, len(STEPS)), dtype=float)
    means, ci_low, ci_high = bootstrap_mean_ci(arr, lower_q=lower_q)
    alive_fraction = [count / total if total else 0.0 for count in alive_counts]

    return {
        "means": means.tolist(),
        "ci_low": ci_low.tolist(),
        "ci_high": ci_high.tolist(),
        "alive_counts": alive_counts,
        "alive_fraction": alive_fraction,
        "total_trajectories": total,
    }


def collect_full_length_trajectory_curve(samples: list[dict], lower_q: float = 0.025) -> dict:
    filtered = [row for row in samples if len(row.get("steps", [])) >= max(STEPS)]
    vectors: list[list[float]] = []
    alive_counts = [0 for _ in STEPS]
    total = 0

    for row in filtered:
        total += 1
        steps = row.get("steps", [])
        vec: list[float] = []
        for idx in STEPS:
            val = steps[idx - 1].get("step_score")
            if val is None:
                val = 0.0
            else:
                alive_counts[idx - 1] += 1
                val = float(val)
            vec.append(val)
        vectors.append(vec)

    arr = np.asarray(vectors, dtype=float) if vectors else np.zeros((0, len(STEPS)), dtype=float)
    means, ci_low, ci_high = bootstrap_mean_ci(arr, lower_q=lower_q)
    alive_fraction = [count / total if total else 0.0 for count in alive_counts]

    return {
        "means": means.tolist(),
        "ci_low": ci_low.tolist(),
        "ci_high": ci_high.tolist(),
        "alive_counts": alive_counts,
        "alive_fraction": alive_fraction,
        "total_trajectories": total,
    }


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


def summarize_model(result_root: Path) -> dict:
    ci_lower_q = 0.025
    for spec in MODEL_SPECS.values():
        if spec["result_root"] == result_root:
            ci_lower_q = float(spec.get("ci_lower_q", 0.025))
            break
    stage_names = sorted([p.name for p in result_root.iterdir() if p.is_dir() and p.name.endswith("_stage")])
    original_samples = load_setting_samples(result_root, stage_names, "Original")
    original_map = {row["id"]: row for row in original_samples if row.get("id")}

    summary: dict[str, dict] = {}
    for setting in SETTING_ORDER:
        if setting == "Original":
            samples = original_samples
        else:
            setting_samples = load_setting_samples(result_root, stage_names, setting)
            setting_map = {row["id"]: row for row in setting_samples if row.get("id")}
            samples = [setting_map.get(sample_id, original_row) for sample_id, original_row in original_map.items()]
        summary[setting] = collect_trajectory_level_curve(samples, lower_q=ci_lower_q)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot true trajectory-level step-score decay for one or more models.")
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
        default=PROJECT_ROOT / "figs" / "interventions" / "true_trajectory_decay",
        help="Output directory for the figure and summary files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="true_trajectory_step_score_decay_zero_padded_multi",
        help="Output filename prefix.",
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
            y = np.array(summary[setting]["means"], dtype=float)
            lo = np.array(summary[setting]["ci_low"], dtype=float)
            hi = np.array(summary[setting]["ci_high"], dtype=float)
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
        ymax = max(max(summary[s]["means"]) for s in SETTING_ORDER)
        ax.set_ylim(0.0, max(0.08, ymax * 1.25))
        ax.tick_params(axis="x", labelbottom=True)

    axes[-1].set_xlabel("Iteration Round", fontsize=18, fontweight="bold")
    axes[0].legend(
        loc="upper center",
        ncol=3,
        frameon=True,
        framealpha=0.96,
        fancybox=True,
        bbox_to_anchor=(0.5, 1.42),
        borderaxespad=0.0,
        handlelength=2.0,
    )
    fig.suptitle(
        "Trajectory-level step-score decay (zero-padded after termination)",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{prefix}.pdf"
    png_path = out_dir / f"{prefix}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def plot_compact_grid(model_summaries: list[tuple[str, dict]], out_dir: Path, prefix: str) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "font.weight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
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

    fig = plt.figure(figsize=(13.8, 7.4))
    gs = fig.add_gridspec(3, 9, height_ratios=[0.18, 1.0, 1.0], hspace=0.32, wspace=0.32)
    axes = [
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
        fig.add_subplot(gs[1, 6:9]),
        fig.add_subplot(gs[2, 1:4]),
        fig.add_subplot(gs[2, 5:8]),
    ]
    x = np.array(STEPS, dtype=float)

    for ax, item in zip(axes, model_summaries):
        model_label, summary = item
        for setting in SETTING_ORDER:
            y = np.array(summary[setting]["means"], dtype=float)
            lo = np.array(summary[setting]["ci_low"], dtype=float)
            hi = np.array(summary[setting]["ci_high"], dtype=float)
            style = SETTING_STYLE[setting]
            ax.fill_between(x, lo, hi, color=style["shadow"], alpha=0.25, linewidth=0)
            ax.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                markersize=6,
                linewidth=2.0,
                label=setting,
            )
        ax.set_title(model_label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ymax = max(max(summary[s]["means"]) for s in SETTING_ORDER)
        ax.set_ylim(0.0, max(0.08, ymax * 1.18))
        ax.set_xlim(0.85, 5.05)
        ax.tick_params(labelsize=10)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            ncol=1,
            frameon=True,
            framealpha=0.95,
            fancybox=True,
            fontsize=8,
        )

    for ax in axes[3:]:
        ax.set_xlabel("Iteration Round", fontsize=12, fontweight="bold")
    for ax in [axes[0], axes[3]]:
        ax.set_ylabel(r"Mean $S_t$", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.935, bottom=0.07, left=0.055, right=0.985)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{prefix}_compact.pdf"
    png_path = out_dir / f"{prefix}_compact.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def write_summary_files(model_summaries: list[tuple[str, dict]], pdf_path: Path, png_path: Path, out_dir: Path, prefix: str) -> tuple[Path, Path]:
    csv_path = out_dir / f"{prefix}_means.csv"
    summary_path = out_dir / f"{prefix}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "setting",
            "step",
            "mean_step_score",
            "ci_low",
            "ci_high",
            "alive_count",
            "alive_fraction",
            "total_trajectories",
        ])
        for model_label, summary in model_summaries:
            for setting in SETTING_ORDER:
                total = summary[setting]["total_trajectories"]
                for idx, step in enumerate(STEPS):
                    writer.writerow([
                        model_label,
                        setting,
                        step,
                        summary[setting]["means"][idx],
                        summary[setting]["ci_low"][idx],
                        summary[setting]["ci_high"][idx],
                        summary[setting]["alive_counts"][idx],
                        summary[setting]["alive_fraction"][idx],
                        total,
                    ])

    payload = {
        model_label: summary
        for model_label, summary in model_summaries
    }
    payload["figure_pdf"] = str(pdf_path)
    payload["figure_png"] = str(png_path)
    payload["csv"] = str(csv_path)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return csv_path, summary_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_summaries: list[tuple[str, dict]] = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        model_summaries.append((spec["label"], summarize_model(spec["result_root"])))

    pdf_path, png_path = plot_multi_model_curves(model_summaries, args.out_dir, args.prefix)
    compact_pdf_path, compact_png_path = plot_compact_grid(model_summaries, args.out_dir, args.prefix)
    csv_path, summary_path = write_summary_files(model_summaries, pdf_path, png_path, args.out_dir, args.prefix)
    print(f"Saved figure: {pdf_path}")
    print(f"Saved image:  {png_path}")
    print(f"Saved compact figure: {compact_pdf_path}")
    print(f"Saved compact image:  {compact_png_path}")
    print(f"Saved table:   {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
