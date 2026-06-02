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
    "Oracle": "first_round_oracle_rewrite/trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
    "Disturb": "first_round_disturb_rewrite_static_prefix_whole/trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
}

SETTING_STYLE = {
    "Original": {"color": "#406D96", "shadow": "#A9C7E6", "marker": "o"},
    "Oracle": {"color": "#2E8B57", "shadow": "#A7D8B8", "marker": "s"},
    "Disturb": {"color": "#C46E2E", "shadow": "#F0C8A2", "marker": "^"},
}

SETTING_ORDER = ["Original", "Oracle", "Disturb"]
GROUPS = [4, 5]


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
        for setting, row_map in [
            ("Original", original_map),
            ("Disturb", disturb_map),
            ("Oracle", oracle_map),
        ]:
            records[setting].append(row_map.get(sample_id, original_row))
    return records


def summarize_model(result_root: Path) -> dict:
    stage_names = sorted([p.name for p in result_root.iterdir() if p.is_dir() and p.name.endswith("_stage")])
    pooled = {setting: {group: [] for group in GROUPS} for setting in SETTING_ORDER}

    for stage_name in stage_names:
        stage_dir = result_root / stage_name
        stage_records = build_stage_full_universe_records(stage_dir)
        for setting in SETTING_ORDER:
            for row in stage_records[setting]:
                steps = row.get("steps", [])
                group = len(steps)
                if group in GROUPS:
                    pooled[setting][group].append(row)

    return pooled


def collect_curve(samples: list[dict]) -> dict:
    vectors: list[list[float]] = []
    total = 0
    alive_counts = [0 for _ in STEPS]

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
                    val = float(val)
                    alive_counts[idx - 1] += 1
            else:
                val = 0.0
            vec.append(val)
        vectors.append(vec)

    arr = np.asarray(vectors, dtype=float) if vectors else np.zeros((0, len(STEPS)), dtype=float)
    means, ci_low, ci_high = bootstrap_mean_ci(arr)
    alive_fraction = [count / total if total else 0.0 for count in alive_counts]

    return {
        "means": means.tolist(),
        "ci_low": ci_low.tolist(),
        "ci_high": ci_high.tolist(),
        "alive_counts": alive_counts,
        "alive_fraction": alive_fraction,
        "total_trajectories": total,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot step-score decay curves for trajectories of length 4 and 5.")
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
        default=PROJECT_ROOT / "figs" / "interventions" / "true_trajectory_decay_len4_len5",
        help="Output directory for the figure and summary files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="true_trajectory_step_score_decay_len4_len5_multi",
        help="Output filename prefix.",
    )
    return parser


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
            "legend.fontsize": 9,
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

    n_models = len(model_summaries)
    fig = plt.figure(figsize=(13.8, 2.8 * n_models + 1.2))
    gs = fig.add_gridspec(n_models, 2, hspace=0.35, wspace=0.18)
    axes = []
    for r in range(n_models):
        for c in range(2):
            axes.append(fig.add_subplot(gs[r, c]))

    x = np.array(STEPS, dtype=float)
    for idx, (model_label, summary) in enumerate(model_summaries):
        row_axes = axes[2 * idx : 2 * idx + 2]
        for ax, group_len in zip(row_axes, GROUPS):
            for setting in SETTING_ORDER:
                curve = summary[setting][group_len]
                y = np.array(curve["means"], dtype=float)
                lo = np.array(curve["ci_low"], dtype=float)
                hi = np.array(curve["ci_high"], dtype=float)
                style = SETTING_STYLE[setting]
                ax.fill_between(x, lo, hi, color=style["shadow"], alpha=0.22, linewidth=0)
                ax.plot(x, y, color=style["color"], marker=style["marker"], markersize=5, linewidth=2.0, label=setting)

            ax.set_title(f"{model_label} | len={group_len}", fontsize=12, fontweight="bold")
            ax.set_xlim(0.85, 5.05)
            ymax = max(max(summary[s][group_len]["means"]) for s in SETTING_ORDER)
            ax.set_ylim(0.0, max(0.08, ymax * 1.2))
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.set_xticks(STEPS)

            if group_len == 4:
                ax.set_ylabel(r"Mean $S_t$", fontsize=12, fontweight="bold")
            if idx == n_models - 1:
                ax.set_xlabel("Iteration Round", fontsize=12, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, framealpha=0.96, bbox_to_anchor=(0.5, 1.01))

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{prefix}_compact.pdf"
    png_path = out_dir / f"{prefix}_compact.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def write_summary(model_summaries: list[tuple[str, dict]], out_dir: Path, prefix: str) -> tuple[Path, Path]:
    csv_path = out_dir / f"{prefix}_summary.csv"
    json_path = out_dir / f"{prefix}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "setting",
                "group_len",
                "num_trajectories",
                "round_1_mean",
                "round_2_mean",
                "round_3_mean",
                "round_4_mean",
                "round_5_mean",
            ]
        )
        payload: dict[str, dict] = {}
        for model_label, summary in model_summaries:
            payload[model_label] = {}
            for setting in SETTING_ORDER:
                payload[model_label][setting] = {}
                for group_len in GROUPS:
                    curve = summary[setting][group_len]
                    payload[model_label][setting][str(group_len)] = curve
                    writer.writerow(
                        [
                            model_label,
                            setting,
                            group_len,
                            curve["total_trajectories"],
                            *[f"{v:.6f}" for v in curve["means"]],
                        ]
                    )

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir

    model_summaries: list[tuple[str, dict]] = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        pooled = summarize_model(spec["result_root"])
        summary: dict[str, dict] = {setting: {} for setting in SETTING_ORDER}
        for setting in SETTING_ORDER:
            for group_len in GROUPS:
                summary[setting][group_len] = collect_curve(pooled[setting][group_len])
        model_summaries.append((spec["label"], summary))

    pdf_path, png_path = plot_compact_grid(model_summaries, out_dir, args.prefix)
    csv_path, json_path = write_summary(model_summaries, out_dir, args.prefix)
    print(f"Saved compact figure: {pdf_path}")
    print(f"Saved compact image:  {png_path}")
    print(f"Saved summary csv:    {csv_path}")
    print(f"Saved summary json:   {json_path}")


if __name__ == "__main__":
    main()
