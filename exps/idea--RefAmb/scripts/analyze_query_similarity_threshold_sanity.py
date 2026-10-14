#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/"
    "crag_mm_2026_03_31_14_06_experiment/trajectory_quality_eval_new/"
    "trajectory_quality_samples.jsonl"
)
OUT_DIR = INPUT_PATH.parent
THRESHOLDS = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def load_text_steps(path: Path) -> list[dict]:
    steps: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            sample_id = sample.get("id")
            for step in sample.get("steps", []):
                if step.get("retrieval_type") != "Text Retrieval":
                    continue
                query_similarity = step.get("query_similarity_with_previous")
                if query_similarity is None:
                    continue
                steps.append(
                    {
                        "sample_id": sample_id,
                        "iteration_index": step.get("iteration_index"),
                        "query_similarity": float(query_similarity),
                        "delta_f": float(step.get("delta_f") or 0.0),
                    }
                )
    return steps


def analyze_thresholds(steps: list[dict], thresholds: list[float]) -> list[dict]:
    rows: list[dict] = []
    for threshold in thresholds:
        triggered = [step for step in steps if step["query_similarity"] > threshold]
        triggered_count = len(triggered)
        triggered_nonzero = sum(1 for step in triggered if step["delta_f"] != 0.0)
        frequency = triggered_nonzero / triggered_count if triggered_count else 0.0
        rows.append(
            {
                "threshold": threshold,
                "triggered_count": triggered_count,
                "triggered_nonzero_delta_f_count": triggered_nonzero,
                "triggered_nonzero_delta_f_frequency": frequency,
            }
        )
    return rows


def plot(rows: list[dict], png_path: Path, pdf_path: Path) -> None:
    style()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)

    points = [(1.0, 0.0)] + [
        (row["threshold"], row["triggered_nonzero_delta_f_frequency"]) for row in rows
    ]
    points = sorted(points, key=lambda item: item[0])
    thresholds = [point[0] for point in points]
    freq = [point[1] for point in points]
    x = np.arange(len(thresholds))

    ax.plot(
        x,
        freq,
        color="#4C78A8",
        marker="o",
        linewidth=2.4,
        markersize=5.5,
        label="Frequency($\\Delta F \\neq 0 \\mid \\mathrm{sim} > \\tau$)",
    )
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_ylim(0.0, max(0.09, max(freq) * 1.2 if freq else 0.09))
    ax.set_xlabel("Query Similarity Threshold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.legend(loc="upper left", frameon=True, fontsize=10.5)
    # ax.set_title("Threshold Sanity Check for Query-Similarity Gating", fontsize=14)
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", colors="#000000")
    ax.xaxis.label.set_color("#000000")
    ax.yaxis.label.set_color("#000000")

    fig.savefig(png_path, dpi=250)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    steps = load_text_steps(INPUT_PATH)
    rows = analyze_thresholds(steps, THRESHOLDS)

    png_path = OUT_DIR / "query_similarity_threshold_sanity.png"
    pdf_path = OUT_DIR / "query_similarity_threshold_sanity.pdf"
    json_path = OUT_DIR / "query_similarity_threshold_sanity.json"

    plot(rows, png_path, pdf_path)

    payload = {
        "input_path": str(INPUT_PATH),
        "num_text_steps_with_similarity": len(steps),
        "threshold_rows": rows,
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
