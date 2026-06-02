#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT_ROOT = Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B")
STAGES = [
    "RefAmb_2026_05_01_11_26_refamb_mcsearch_stage",
    "RefAmb_2026_05_01_13_28_refamb_oven_stage",
    "RefAmb_2026_05_02_13_50_refamb_infoseek_stage",
    "RefAmb_2026_05_02_14_22_refamb_crag_stage",
]

SETTING_PATHS = {
    "Original": "trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
    "Oracle": "first_round_oracle_rewrite/trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
    "Disturb": "first_round_disturb_rewrite_static_prefix/trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl",
}

OUT_DIR = Path("/home/you/FlashRAG/exps/idea10/scripts")
TABLE_TEX = OUT_DIR / "refamb_stagnation_table.tex"
FIG_PDF = OUT_DIR / "refamb_delta_f_stagnation_curve.pdf"
FIG_PNG = OUT_DIR / "refamb_delta_f_stagnation_curve.png"
JSON_OUT = OUT_DIR / "refamb_stagnation_summary.json"

BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 17


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_path(stage_dir: Path, rel_path: str) -> Path:
    path = stage_dir / rel_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_setting_samples(setting: str) -> list[dict]:
    samples: list[dict] = []
    rel_path = SETTING_PATHS[setting]
    for stage in STAGES:
        stage_dir = RESULT_ROOT / stage
        path = resolve_path(stage_dir, rel_path)
        samples.extend(list(load_jsonl(path)))
    return samples


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


def build_iteration_stats(samples: list[dict]) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in samples:
        for step in row.get("steps", []):
            idx = step.get("iteration_index")
            if idx is None:
                continue
            idx = int(idx)
            if not 1 <= idx <= 5:
                continue

            sim = step.get("query_similarity_with_previous")
            if sim is not None:
                stats[idx]["sim_total"] += 1.0
                if abs(float(sim) - 1.0) < 1e-12:
                    stats[idx]["sim_eq1"] += 1.0

            delta_f = step.get("delta_f")
            if delta_f is not None:
                df = float(delta_f)
                stats[idx]["df_total"] += 1.0
                stats[idx]["df0"] += 1.0 if abs(df) < 1e-12 else 0.0
                stats[idx].setdefault("delta_values", []).append(df)

    final_stats: dict[int, dict[str, float]] = {}
    for idx in range(1, 6):
        raw = stats.get(idx, {})
        delta_values = raw.get("delta_values", [])
        mean_df, ci_low, ci_high = bootstrap_mean_ci(delta_values)
        final_stats[idx] = {
            "sim_total": float(raw.get("sim_total", 0.0)),
            "sim_eq1": float(raw.get("sim_eq1", 0.0)),
            "sim_eq1_ratio": float(raw.get("sim_eq1", 0.0) / raw.get("sim_total", 0.0)) if raw.get("sim_total", 0.0) else 0.0,
            "df_total": float(raw.get("df_total", 0.0)),
            "df0": float(raw.get("df0", 0.0)),
            "df0_ratio": float(raw.get("df0", 0.0) / raw.get("df_total", 0.0)) if raw.get("df_total", 0.0) else 0.0,
            "mean_delta_f": mean_df,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return final_stats


def build_table_tex(all_stats: dict[str, dict[int, dict[str, float]]]) -> str:
    def global_counts(setting_stats: dict[int, dict[str, float]]) -> dict[str, float]:
        sim_eq1 = sum(row["sim_eq1"] for row in setting_stats.values())
        sim_total = sum(row["sim_total"] for row in setting_stats.values())
        df0 = sum(row["df0"] for row in setting_stats.values())
        df_total = sum(row["df_total"] for row in setting_stats.values())
        mean_delta = mean(row["mean_delta_f"] for row in setting_stats.values())
        return {
            "sim_eq1": sim_eq1,
            "sim_total": sim_total,
            "df0": df0,
            "df_total": df_total,
            "sim_ratio": sim_eq1 / sim_total if sim_total else 0.0,
            "df_ratio": df0 / df_total if df_total else 0.0,
            "mean_delta": mean_delta,
        }

    global_rows = {setting: global_counts(stats) for setting, stats in all_stats.items()}
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"Setting & Query similarity $=1$ & $\Delta F=0$ & Mean $\Delta F$ \\")
    lines.append(r"\hline")
    for setting in ["Original", "Oracle", "Disturb"]:
        row = global_rows[setting]
        lines.append(
            f"{setting} & "
            f"{row['sim_ratio']*100:.1f}\\% ({int(row['sim_eq1'])}/{int(row['sim_total'])}) & "
            f"{row['df_ratio']*100:.1f}\\% ({int(row['df0'])}/{int(row['df_total'])}) & "
            f"{row['mean_delta']:.4f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Across four sources and all iterations, the overall ratio of repeated queries (query similarity with previous query equals 1) and zero-information steps ($\Delta F=0$), together with the mean $\Delta F$. This summarizes the global stagnation pattern.}"
    )
    lines.append(r"\label{tab:refamb-stagnation-ratios}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def plot_delta_f_curve(all_stats: dict[str, dict[int, dict[str, float]]]) -> None:
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
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    styles = {
        "Original": {"color": "#406D96", "shadow": "#A9C7E6", "marker": "o"},
        "Oracle": {"color": "#2E8B57", "shadow": "#A7D8B8", "marker": "s"},
        "Disturb": {"color": "#C46E2E", "shadow": "#F0C8A2", "marker": "^"},
    }
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    for setting in ["Original", "Oracle", "Disturb"]:
        mean_y = np.array([all_stats[setting][i]["mean_delta_f"] for i in range(1, 6)], dtype=float)
        ci_low = np.array([all_stats[setting][i]["ci_low"] for i in range(1, 6)], dtype=float)
        ci_high = np.array([all_stats[setting][i]["ci_high"] for i in range(1, 6)], dtype=float)
        ax.fill_between(x, ci_low, ci_high, color=styles[setting]["shadow"], alpha=0.35, linewidth=0)
        ax.plot(
            x,
            mean_y,
            color=styles[setting]["color"],
            marker=styles[setting]["marker"],
            markersize=4.5,
            linewidth=1.8,
            label=setting,
        )

    ax.set_xlabel("Iteration Round")
    ax.set_ylabel(r"Mean $\Delta F$")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.85, 5.15)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_linewidth(0.8)
    ax.spines["right"].set_linewidth(0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_color("black")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {setting: build_iteration_stats(load_setting_samples(setting)) for setting in SETTING_PATHS}

    TABLE_TEX.write_text(build_table_tex(all_stats), encoding="utf-8")
    plot_delta_f_curve(all_stats)

    payload = {
        "settings": {
            setting: {
                str(idx): {k: v for k, v in stats.items() if k != "delta_values"}
                for idx, stats in per_iter.items()
            }
            for setting, per_iter in all_stats.items()
        },
        "table_tex": str(TABLE_TEX),
        "figure_pdf": str(FIG_PDF),
        "figure_png": str(FIG_PNG),
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
