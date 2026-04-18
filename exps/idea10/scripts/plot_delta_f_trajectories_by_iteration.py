#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH_1 = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/trajectory_quality_eval/"
    "trajectory_quality_samples.jsonl"
)
INPUT_PATH_2 = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_04_16_10_21_api_experiment/trajectory_quality_eval/"
    "trajectory_quality_samples.jsonl"
)
OUT_DIR = INPUT_PATH_1.parent
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
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)

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
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=item["label"],
            zorder=2,
        )

    ax.set_xlabel("Iteration Round", fontsize=13)
    ax.set_ylabel("Delta F", fontsize=13)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.75, 5.25)
    ax.set_ylim(bottom=-0.01)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.55)
    ax.legend(loc="upper right", frameon=True, fontsize=10.5)
    # ax.set_title("Delta F Decays Rapidly Across Iterations", fontsize=14.5)

    fig.savefig(png_path, dpi=250)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectories_1 = load_trajectories(INPUT_PATH_1)
    trajectories_2 = load_trajectories(INPUT_PATH_2)

    png_path = OUT_DIR / "delta_f_trajectories_by_iteration.png"
    pdf_path = OUT_DIR / "delta_f_trajectories_by_iteration.pdf"
    json_path = OUT_DIR / "delta_f_trajectories_by_iteration.json"

    summary_1 = build_summary(trajectories_1)
    summary_2 = build_summary(trajectories_2)
    plot(
        [
            {
                "label": "CRAG-MM Mar 31",
                "line_color": "#D62828",
                "shadow_color": "#F4A6A6",
                "summary": summary_1,
            },
            {
                "label": "CRAG-MM Apr 16 API",
                "line_color": "#1D4ED8",
                "shadow_color": "#93C5FD",
                "summary": summary_2,
            },
        ],
        png_path,
        pdf_path,
    )

    payload = {
        "input_paths": [str(INPUT_PATH_1), str(INPUT_PATH_2)],
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "paper_style": {
            "top_panel_style": "two_mean_lines_with_shadow_ci_only",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "series": [
            {"label": "Qwen2.5-VL-7B-Instruct", "summary": summary_1},
            {"label": "Qwen2.5-VL-72B-Instruct", "summary": summary_2},
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
