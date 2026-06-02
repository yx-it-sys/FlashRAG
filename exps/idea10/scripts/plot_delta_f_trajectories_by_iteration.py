#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULT_ROOT = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B"
)
INPUT_PATHS_7B = [
    RESULT_ROOT
    / "RefAmb_2026_05_01_11_26_refamb_mcsearch_stage/trajectory_quality_eval_whole_delta_F/trajectory_quality_samples.jsonl",
    RESULT_ROOT
    / "RefAmb_2026_05_01_13_28_refamb_oven_stage/trajectory_quality_eval_whole_delta_F/trajectory_quality_samples.jsonl",
    RESULT_ROOT
    / "RefAmb_2026_05_02_13_50_refamb_infoseek_stage/trajectory_quality_eval_whole_delta_F/trajectory_quality_samples.jsonl",
    RESULT_ROOT
    / "RefAmb_2026_05_02_14_22_refamb_crag_stage/trajectory_quality_eval_whole_delta_F/trajectory_quality_samples.jsonl",
]
# Qwen2.5-VL-72B-Instruct results are not available for this refreshed plot yet.
# INPUT_PATHS_72B = []
OUT_DIR = RESULT_ROOT / "stats"
BOOTSTRAP_SAMPLES = 2000


def load_trajectories(path: Path) -> list[dict]:
    trajectories: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            by_iteration = {idx: 0.0 for idx in range(1, 6)}
            for step in sample.get("steps", []):
                iteration_index = step.get("iteration_index")
                if iteration_index is None:
                    continue
                iteration_index = int(iteration_index)
                if iteration_index < 1 or iteration_index > 5:
                    continue
                delta_f = step.get("delta_f")
                by_iteration[iteration_index] = float(delta_f) if delta_f is not None else 0.0
            trajectories.append(
                {
                    "id": sample.get("id"),
                    "question": sample.get("question"),
                    "x": [1, 2, 3, 4, 5],
                    "y": [by_iteration[idx] for idx in range(1, 6)],
                }
            )
    return trajectories


def load_trajectories_from_paths(paths: list[Path]) -> list[dict]:
    trajectories: list[dict] = []
    for path in paths:
        trajectories.extend(load_trajectories(path))
    return trajectories


def compute_round_matrix(trajectories: list[dict]) -> np.ndarray:
    return np.array([traj["y"] for traj in trajectories], dtype=float)


def bootstrap_mean_ci(
    values: np.ndarray, num_bootstrap: int = BOOTSTRAP_SAMPLES
) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    n = values.size
    bootstrap_means = np.empty(num_bootstrap, dtype=float)
    for idx in range(num_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means[idx] = sample.mean()
    return (
        float(values.mean()),
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    )


def build_summary(trajectories: list[dict]) -> dict:
    matrix = compute_round_matrix(trajectories)
    round_stats = []
    for round_idx in range(1, 6):
        values = matrix[:, round_idx - 1]
        mean_delta_f, ci_low, ci_high = bootstrap_mean_ci(values)
        round_stats.append(
            {
                "iteration_round": round_idx,
                "count": int(values.size),
                "mean_delta_f": mean_delta_f,
                "median_delta_f": float(np.median(values)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "nonzero_rate": float((values > 0).mean()),
                "max_delta_f": float(values.max()),
                "min_delta_f": float(values.min()),
            }
        )
    return {"num_items": len(trajectories), "round_stats": round_stats}


def plot(series: list[dict], png_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax_top = ax.twiny()

    x_rounds = np.array([1, 2, 3, 4, 5], dtype=float)
    for item in series:
        summary = item["summary"]
        mean_y = np.array([row["mean_delta_f"] for row in summary["round_stats"]], dtype=float)
        ci_low = np.array([row["ci95_low"] for row in summary["round_stats"]], dtype=float)
        ci_high = np.array([row["ci95_high"] for row in summary["round_stats"]], dtype=float)

        ax.fill_between(
            x_rounds,
            ci_low,
            ci_high,
            color=item["shadow_color"],
            alpha=0.28,
            label="_nolegend_",
            zorder=1,
        )
        ax.plot(
            x_rounds,
            mean_y,
            color=item["line_color"],
            linewidth=1.4,
            marker=item["marker"],
            markersize=item["markersize"],
            label=item["label"],
            zorder=2,
        )

    ax.set_xlabel("Iteration Round")
    ax.set_ylabel(r"$\Delta\mathcal{F}$")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.75, 5.25)
    ax.set_ylim(0.0, 1.00)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks([])
    ax_top.set_xlabel("")
    ax_top.spines["top"].set_visible(True)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", top=False, bottom=False, labeltop=False)

    ax.legend(loc="upper right", frameon=False, ncol=1)

    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectories_1 = load_trajectories_from_paths(INPUT_PATHS_7B)

    png_path = OUT_DIR / "delta_f_trajectories_by_iteration.png"
    pdf_path = OUT_DIR / "delta_f_trajectories_by_iteration.pdf"
    json_path = OUT_DIR / "delta_f_trajectories_by_iteration.json"

    summary_1 = build_summary(trajectories_1)
    plot(
        [
            {
                "label": "Qwen2.5-VL-7B-Instruct",
                "line_color": "#1f77b4",
                "shadow_color": "#A9C8EB",
                "marker": "s",
                "markersize": 3.8,
                "summary": summary_1,
            },
        ],
        png_path,
        pdf_path,
    )

    payload = {
        "input_paths": [str(path) for path in INPUT_PATHS_7B],
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "paper_style": {
            "top_panel_style": "two_mean_lines_with_shadow_ci_only",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "series": [
            {"label": "Qwen2.5-VL-7B-Instruct", "summary": summary_1},
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
