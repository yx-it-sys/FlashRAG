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
    pooled = {setting: [] for setting in SETTING_ORDER}
    for stage_name in stage_names:
        stage_dir = result_root / stage_name
        stage_records = build_stage_full_universe_records(stage_dir)
        for setting in SETTING_ORDER:
            pooled[setting].extend(stage_records[setting])
    return pooled


def collect_alive_only_curve(records: list[dict]) -> dict:
    means: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []
    alive_counts: list[int] = []
    total = len(records)

    for step in STEPS:
        values: list[float] = []
        count = 0
        for row in records:
            steps = row.get("steps", [])
            if len(steps) >= step:
                val = steps[step - 1].get("step_score")
                if val is not None:
                    values.append(float(val))
                    count += 1
        alive_counts.append(count)
        if values:
            arr = np.asarray(values, dtype=float)
            m = float(arr.mean())
            if arr.size == 1:
                lo = hi = m
            else:
                rng = np.random.default_rng(BOOTSTRAP_SEED)
                idx = rng.integers(0, arr.size, size=(BOOTSTRAP_SAMPLES, arr.size))
                boot_means = arr[idx].mean(axis=1)
                lo = float(np.quantile(boot_means, 0.025))
                hi = float(np.quantile(boot_means, 0.975))
        else:
            m = lo = hi = 0.0
        means.append(m)
        ci_low.append(lo)
        ci_high.append(hi)

    alive_fraction = [c / total if total else 0.0 for c in alive_counts]
    return {
        "means": means,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alive_counts": alive_counts,
        "alive_fraction": alive_fraction,
        "total_trajectories": total,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot alive-only trajectory decay curves for one or more models.")
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
        default=PROJECT_ROOT / "figs" / "interventions" / "true_trajectory_decay_alive_only",
        help="Output directory for the figure and summary files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="true_trajectory_step_score_decay_alive_only_multi",
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

    if len(model_summaries) == 5:
        fig = plt.figure(figsize=(13.8, 7.4))
        gs = fig.add_gridspec(3, 9, height_ratios=[0.18, 1.0, 1.0], hspace=0.32, wspace=0.32)
        axes = [
            fig.add_subplot(gs[1, 0:3]),
            fig.add_subplot(gs[1, 3:6]),
            fig.add_subplot(gs[1, 6:9]),
            fig.add_subplot(gs[2, 1:4]),
            fig.add_subplot(gs[2, 5:8]),
        ]
    else:
        n_rows = len(model_summaries)
        fig = plt.figure(figsize=(13.8, 2.8 * n_rows + 1.2))
        gs = fig.add_gridspec(n_rows, 1, hspace=0.35)
        axes = [fig.add_subplot(gs[r, 0]) for r in range(n_rows)]
    x = np.array(STEPS, dtype=float)

    for ax, (model_label, summary) in zip(axes, model_summaries):
        for setting in SETTING_ORDER:
            curve = summary[setting]
            y = np.array(curve["means"], dtype=float)
            lo = np.array(curve["ci_low"], dtype=float)
            hi = np.array(curve["ci_high"], dtype=float)
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
        ax.set_xlim(0.85, 5.05)
        ymax = max(max(summary[s]["means"]) for s in SETTING_ORDER)
        ax.set_ylim(0.0, max(0.08, ymax * 1.18))
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_xticks(STEPS)
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

    if len(model_summaries) == 5:
        for ax in axes[3:]:
            ax.set_xlabel("Iteration Round", fontsize=12, fontweight="bold")
        for ax in [axes[0], axes[3]]:
            ax.set_ylabel(r"Mean $S_t$", fontsize=12, fontweight="bold")
        fig.subplots_adjust(top=0.935, bottom=0.07, left=0.055, right=0.985)
    else:
        axes[-1].set_xlabel("Iteration Round", fontsize=12, fontweight="bold")
        fig.subplots_adjust(top=0.94, bottom=0.08, left=0.055, right=0.985)

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
                "step",
                "mean_step_score",
                "ci_low",
                "ci_high",
                "alive_count",
                "alive_fraction",
                "total_trajectories",
            ]
        )
        payload: dict[str, dict] = {}
        for model_label, summary in model_summaries:
            payload[model_label] = summary
            for setting in SETTING_ORDER:
                total = summary[setting]["total_trajectories"]
                for idx, step in enumerate(STEPS):
                    writer.writerow(
                        [
                            model_label,
                            setting,
                            step,
                            summary[setting]["means"][idx],
                            summary[setting]["ci_low"][idx],
                            summary[setting]["ci_high"][idx],
                            summary[setting]["alive_counts"][idx],
                            summary[setting]["alive_fraction"][idx],
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
        summary: dict[str, dict] = {}
        for setting in SETTING_ORDER:
            summary[setting] = collect_alive_only_curve(pooled[setting])
        model_summaries.append((spec["label"], summary))

    pdf_path, png_path = plot_compact_grid(model_summaries, out_dir, args.prefix)
    csv_path, json_path = write_summary(model_summaries, out_dir, args.prefix)
    print(f"Saved compact figure: {pdf_path}")
    print(f"Saved compact image:  {png_path}")
    print(f"Saved summary csv:    {csv_path}")
    print(f"Saved summary json:   {json_path}")


if __name__ == "__main__":
    main()
