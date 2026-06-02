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
from matplotlib.lines import Line2D


STEPS = [1, 2, 3, 4, 5]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_SEED = 17

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
    "Original": {"color": "#406D96", "fit": "#234C77"},
    "Oracle": {"color": "#2E8B57", "fit": "#1D5B39"},
    "Disturb": {"color": "#C46E2E", "fit": "#8B4A17"},
}

SETTING_ORDER = ["Original", "Oracle", "Disturb"]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def summarize_model(result_root: Path) -> dict[str, list[dict]]:
    stage_names = sorted([p.name for p in result_root.iterdir() if p.is_dir() and p.name.endswith("_stage")])
    pooled = {setting: [] for setting in SETTING_ORDER}
    for stage_name in stage_names:
        stage_dir = result_root / stage_name
        stage_records = build_stage_full_universe_records(stage_dir)
        for setting in SETTING_ORDER:
            pooled[setting].extend(stage_records[setting])
    return pooled


def collect_observed_round_means(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    buckets = {step: [] for step in STEPS}
    for row in records:
        steps = row.get("steps", [])
        for step in STEPS:
            if len(steps) >= step:
                val = steps[step - 1].get("step_score")
                if val is not None:
                    buckets[step].append(float(val))

    means = np.array([float(np.mean(buckets[step])) if buckets[step] else 0.0 for step in STEPS], dtype=float)
    counts = np.array([len(buckets[step]) for step in STEPS], dtype=int)
    return means, counts


def fit_curve_from_round_means(means: np.ndarray) -> np.ndarray:
    x = np.asarray(STEPS, dtype=float)
    degree = min(3, len(STEPS) - 1)
    coeff = np.polyfit(x, means, degree)
    x_dense = np.linspace(float(x.min()), float(x.max()), 200)
    y_dense = np.polyval(coeff, x_dense)
    return x_dense, y_dense, coeff


def extract_item_curves(records: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for row in records:
        steps = row.get("steps", [])
        if not steps:
            continue
        xs = []
        ys = []
        for idx, step in enumerate(steps, start=1):
            val = step.get("step_score")
            if val is None:
                continue
            xs.append(float(idx))
            ys.append(float(val))
        if xs:
            curves.append((np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
    return curves


def plot_model(model_label: str, model_summary: dict[str, list[dict]], out_dir: Path, prefix: str) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
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

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.2), sharey=True, sharex=True)
    y_max = 0.0
    for setting in SETTING_ORDER:
        for row in model_summary[setting]:
            steps = row.get("steps", [])
            if not steps:
                continue
            local_max = max(float(step.get("step_score", 0.0) or 0.0) for step in steps)
            y_max = max(y_max, local_max)
    y_max = max(0.08, y_max * 1.18)

    for ax, setting in zip(axes, SETTING_ORDER):
        style = SETTING_STYLE[setting]
        records = model_summary[setting]
        curves = extract_item_curves(records)
        for xs, ys in curves:
            ax.plot(xs, ys, color=style["color"], alpha=0.07, linewidth=0.9)

        means, counts = collect_observed_round_means(records)
        x_dense, y_dense, coeff = fit_curve_from_round_means(means)
        ax.plot(x_dense, y_dense, color=style["fit"], linewidth=3.0, label="Fitted curve")
        ax.scatter(STEPS, means, color="black", s=22, zorder=4, label="Round mean")

        ax.set_title(setting, fontsize=13, fontweight="bold")
        ax.set_xlim(0.85, 5.05)
        ax.set_ylim(0.0, y_max)
        ax.grid(axis="y", linestyle="--", alpha=0.30)
        ax.set_xticks(STEPS)
        ax.text(
            0.03,
            0.94,
            f"n={len(records)}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        ax.legend(
            handles=[
                Line2D([0], [0], color=style["color"], alpha=0.20, lw=3, label="Item trajectories"),
                Line2D([0], [0], color=style["fit"], lw=3, label="Fitted curve"),
                Line2D([0], [0], marker="o", color="black", lw=0, markersize=5, label="Round mean"),
            ],
            loc="upper right",
            frameon=True,
            framealpha=0.95,
        )

    axes[0].set_ylabel("Step Score", fontsize=13, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Iteration Round", fontsize=13, fontweight="bold")
    fig.suptitle(model_label, fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{prefix}_{model_label}.pdf"
    png_path = out_dir / f"{prefix}_{model_label}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def write_summary(model_summaries: dict[str, dict[str, list[dict]]], out_dir: Path, prefix: str) -> tuple[Path, Path]:
    csv_path = out_dir / f"{prefix}_summary.csv"
    json_path = out_dir / f"{prefix}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "setting",
                "num_trajectories",
                "round_1_mean",
                "round_2_mean",
                "round_3_mean",
                "round_4_mean",
                "round_5_mean",
                "poly_degree",
                "poly_coefficients",
            ]
        )
        payload: dict[str, dict] = {}
        for model_label, summary in model_summaries.items():
            payload[model_label] = {}
            for setting in SETTING_ORDER:
                means, _ = collect_observed_round_means(summary[setting])
                degree = min(3, len(STEPS) - 1)
                coeff = np.polyfit(np.asarray(STEPS, dtype=float), means, degree)
                payload[model_label][setting] = {
                    "num_trajectories": len(summary[setting]),
                    "round_means": means.tolist(),
                    "poly_degree": degree,
                    "poly_coefficients": coeff.tolist(),
                }
                writer.writerow(
                    [
                        model_label,
                        setting,
                        len(summary[setting]),
                        *[f"{v:.6f}" for v in means],
                        degree,
                        json.dumps([float(v) for v in coeff.tolist()]),
                    ]
                )

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot all item trajectories with fitted decay curves.")
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
        default=PROJECT_ROOT / "figs" / "interventions" / "trajectory_fit",
        help="Output directory for figures and summaries.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="true_trajectory_spaghetti_fit_multi",
        help="Filename prefix for outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir

    model_summaries: dict[str, dict[str, list[dict]]] = {}
    pdf_paths: list[Path] = []
    png_paths: list[Path] = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        summary = summarize_model(spec["result_root"])
        model_label = spec["label"]
        model_summaries[model_label] = summary
        pdf_path, png_path = plot_model(model_label, summary, out_dir, args.prefix)
        pdf_paths.append(pdf_path)
        png_paths.append(png_path)
        print(f"Saved figure: {pdf_path}")
        print(f"Saved image:  {png_path}")

    csv_path, json_path = write_summary(model_summaries, out_dir, args.prefix)
    print(f"Saved summary csv: {csv_path}")
    print(f"Saved summary json: {json_path}")


if __name__ == "__main__":
    main()
