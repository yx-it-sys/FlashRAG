#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors


BASE_DIR = Path("/home/you/FlashRAG/exps/idea10/idea_reports/papers")
RAW_DIR = BASE_DIR / "raw_data"
FIG_DIR = BASE_DIR / "figs"
RESULT_ROOT = Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B")
STAGE_DIRS = [
    "RefAmb_2026_05_01_11_26_refamb_mcsearch_stage",
    "RefAmb_2026_05_01_13_28_refamb_oven_stage",
    "RefAmb_2026_05_02_13_50_refamb_infoseek_stage",
    "RefAmb_2026_05_02_14_22_refamb_crag_stage",
]
SETTING_SCORE_SUBDIR = {
    "Original": "",
    "Disturb": "first_round_disturb_rewrite_static_prefix_whole",
    "Oracle": "first_round_oracle_rewrite",
}

SPLITS = [
    ("correct", RAW_DIR / "trajectory_quality_by_task_type_original_correct.csv"),
    ("incorrect", RAW_DIR / "trajectory_quality_by_task_type_original_incorrect.csv"),
]

SETTINGS = ["Original", "Disturb", "Oracle"]
TASK_ORDER = [
    "E.R.",
    "S.A.Q.",
    "M.H.",
    "Comp.",
    "S.A.",
    "Avg.",
]

# Map axis labels back to the task_type values stored in the CSV files.
TASK_NAME_MAP = {
    "E.R.": "Entity Recognition",
    "S.A.Q.": "Single-hop Attribute Query",
    "M.H.": "Multi-hop",
    "Comp.": "Comparison",
    "S.A.": "Subproblem Aggregation",
    "Avg.": "Macro-average",
}

METRICS = [
    ("rte", "RTE"),
    ("iter", "#Iter."),
    ("delta_f_avg", r"$\Delta F_{avg}$"),
    ("u_avg", r"$U_{avg}$"),
    ("lj_score", "LJ score"),
]

MAIN_METRICS = [
    ("rte", "RTE"),
    ("lj_score", "LJ score"),
    ("flip_rate", "Flip ratio"),
]

APPENDIX_METRICS = [
    ("iter", "#Iter."),
    ("delta_f_avg", r"$\Delta F_{avg}$"),
    ("u_avg", r"$U_{avg}$"),
]

N_BINS = 30
BIN_GAMMA = 3.0

SETTING_COLUMNS = {
    "Original": {
        "rte": "Original_rte",
        "iter": "Original_iter",
        "delta_f_avg": "Original_delta_f_avg",
        "u_avg": "Original_u_avg",
        "lj_score": "Original_lj_score",
    },
    "Disturb": {
        "rte": "Disturb_rte",
        "iter": "Disturb_iter",
        "delta_f_avg": "Disturb_delta_f_avg",
        "u_avg": "Disturb_u_avg",
        "lj_score": "Disturb_lj_score",
    },
    "Oracle": {
        "rte": "Oracle_rte",
        "iter": "Oracle_iter",
        "delta_f_avg": "Oracle_delta_f_avg",
        "u_avg": "Oracle_u_avg",
        "lj_score": "Oracle_lj_score",
    },
}

CSV_FLOAT_COLUMNS = {col for setting in SETTING_COLUMNS.values() for col in setting.values()}


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str) -> float:
    if value is None or value == "" or value == "--":
        return math.nan
    return float(value)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_score_map(stage_dir: Path, setting: str) -> dict[str, dict]:
    subdir = SETTING_SCORE_SUBDIR[setting]
    score_path = stage_dir / "gpt_acc_score.json" if not subdir else stage_dir / subdir / "gpt_acc_score.json"
    rows = load_json(score_path)
    return {row["id"]: row for row in rows if row.get("id")}


def is_one(value: float) -> bool:
    return abs(value - 1.0) < 1e-12


def is_zero(value: float) -> bool:
    return abs(value - 0.0) < 1e-12


def compute_flip_rate_by_split() -> dict[str, dict[str, dict[str, float]]]:
    numerators: dict[str, dict[str, dict[str, int]]] = {
        "correct": defaultdict(lambda: defaultdict(int)),
        "incorrect": defaultdict(lambda: defaultdict(int)),
    }
    denominators: dict[str, dict[str, dict[str, int]]] = {
        "correct": defaultdict(lambda: defaultdict(int)),
        "incorrect": defaultdict(lambda: defaultdict(int)),
    }

    for stage_name in STAGE_DIRS:
        stage_dir = RESULT_ROOT / stage_name
        original_map = load_score_map(stage_dir, "Original")
        disturb_map = load_score_map(stage_dir, "Disturb")
        oracle_map = load_score_map(stage_dir, "Oracle")

        for sample_id, original_item in original_map.items():
            task_type = original_item.get("task_type")
            if task_type not in TASK_NAME_MAP.values():
                continue
            original_acc = float(original_item["output"]["metric_score"]["gpt_acc"])

            for split_name, should_include in [
                ("correct", is_one(original_acc)),
                ("incorrect", is_zero(original_acc)),
            ]:
                if not should_include:
                    continue
                for setting, setting_map in [("Disturb", disturb_map), ("Oracle", oracle_map)]:
                    metric_item = setting_map.get(sample_id)
                    if metric_item is None:
                        continue
                    target_acc = float(metric_item["output"]["metric_score"]["gpt_acc"])
                    denominators[split_name][task_type][setting] += 1

                    if split_name == "incorrect":
                        flipped = is_one(target_acc)
                    else:
                        flipped = is_zero(target_acc)
                    if flipped:
                        numerators[split_name][task_type][setting] += 1

    out: dict[str, dict[str, dict[str, float]]] = {"correct": {}, "incorrect": {}}
    for split_name in ["correct", "incorrect"]:
        task_rates: dict[str, dict[str, float]] = {}
        for task_type in TASK_NAME_MAP.values():
            task_rates[task_type] = {"Original": math.nan}
            for setting in ["Disturb", "Oracle"]:
                denom = denominators[split_name][task_type][setting]
                if denom == 0:
                    task_rates[task_type][setting] = math.nan
                else:
                    task_rates[task_type][setting] = numerators[split_name][task_type][setting] / denom
        macro = {"Original": math.nan}
        for setting in ["Disturb", "Oracle"]:
            vals = [
                task_rates[task][setting]
                for task in TASK_NAME_MAP.values()
                if not math.isnan(task_rates[task][setting])
            ]
            macro[setting] = math.nan if not vals else sum(vals) / len(vals)
        task_rates["Macro-average"] = macro
        out[split_name] = task_rates
    return out


def build_matrix(
    rows: list[dict],
    metric_key: str,
    split_name: str,
    flip_rate_by_split: dict[str, dict[str, dict[str, float]]],
) -> list[list[float]]:
    if metric_key == "flip_rate":
        split_rates = flip_rate_by_split[split_name]
        matrix: list[list[float]] = []
        for task_label in TASK_ORDER:
            task_name = TASK_NAME_MAP.get(task_label, task_label)
            matrix.append([split_rates[task_name].get(setting, math.nan) for setting in SETTINGS])
        return matrix

    row_by_task = {row["task_type"]: row for row in rows}
    matrix: list[list[float]] = []
    for task_label in TASK_ORDER:
        task_name = TASK_NAME_MAP.get(task_label, task_label)
        row = row_by_task[task_name]
        matrix.append([parse_float(row[SETTING_COLUMNS[setting][metric_key]]) for setting in SETTINGS])
    return matrix


def matrix_range(matrix: list[list[float]]) -> tuple[float, float]:
    values = [value for row in matrix for value in row if not math.isnan(value)]
    if not values:
        return 0.0, 1.0
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return vmin, vmin + 1e-6
    return vmin, vmax


def annotate_heatmap(ax, matrix, norm, vmin: float, vmax: float):
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if math.isnan(value):
                text = "--"
                color = "black"
            else:
                text = f"{value:.3f}"
                color = "white" if value >= (vmin + vmax) / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=14, color=color)


def draw_panel(ax, matrix, metric_name: str, show_ylabel: bool, xlabels: list[str]):
    vmin, vmax = matrix_range(matrix)
    # Use non-uniform bins so higher values get finer color resolution.
    # This makes differences such as 0.845 vs 0.723 map to more distinct blocks.
    t = [(i / N_BINS) ** BIN_GAMMA for i in range(N_BINS + 1)]
    boundaries = [vmin + (vmax - vmin) * x for x in t]
    cmap = plt.colormaps["YlGnBu"].resampled(N_BINS)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=0)
    ax.set_yticks(range(len(TASK_ORDER)))
    if show_ylabel:
        ax.set_yticklabels(TASK_ORDER)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0)
    ax.set_title(metric_name, pad=6)
    annotate_heatmap(ax, matrix, norm, vmin, vmax)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([x - 0.5 for x in range(len(xlabels) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(TASK_ORDER) + 1)], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def draw_split_panels(
    subfig,
    split_name: str,
    rows: list[dict],
    metrics: list[tuple[str, str]],
    split_title: str,
    show_ylabel: bool,
    flip_rate_by_split: dict[str, dict[str, dict[str, float]]],
):
    axes = subfig.subplots(1, len(metrics), squeeze=False)
    axes = axes[0]

    for idx, (metric_key, metric_name) in enumerate(metrics):
        matrix = build_matrix(rows, metric_key, split_name, flip_rate_by_split)
        xlabels = SETTINGS
        if metric_key == "flip_rate":
            matrix = [row[1:] for row in matrix]
            xlabels = ["Disturb", "Oracle"]
        draw_panel(axes[idx], matrix, metric_name, show_ylabel=(show_ylabel and idx == 0), xlabels=xlabels)

    # Place the split label underneath the subfigure so it reads like a caption.
    subfig.text(0.5, -0.06, split_title, ha="center", va="top", fontsize=28)


def save_heatmap(
    split_rows: list[tuple[str, list[dict]]],
    metrics: list[tuple[str, str]],
    suffix: str,
    orientation: str,
    flip_rate_by_split: dict[str, dict[str, dict[str, float]]],
) -> Path:
    plt.rcParams.update(
        {
            "font.size": 55,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 25,
            "axes.labelsize": 25,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    if orientation == "horizontal":
        fig_width = max(17.0, 4.4 * len(metrics) * len(split_rows))
        fig_height = 5.8
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        subfigs = fig.subfigures(1, len(split_rows), wspace=0.05)
        if len(split_rows) == 1:
            subfigs = [subfigs]
    elif orientation == "vertical":
        fig_width = max(8.8, 4.4 * len(metrics))
        fig_height = max(10.4, 5.2 * len(split_rows))
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        subfigs = fig.subfigures(len(split_rows), 1, hspace=0.12)
        if len(split_rows) == 1:
            subfigs = [subfigs]
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")

    for idx, (subfig, (split_name, rows)) in enumerate(zip(subfigs, split_rows)):
        show_ylabel = orientation == "vertical" or idx == 0
        draw_split_panels(
            subfig,
            split_name,
            rows,
            metrics,
            f"Original-{split_name}",
            show_ylabel=show_ylabel,
            flip_rate_by_split=flip_rate_by_split,
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"refamb_trajectory_quality_original_{suffix}_heatmap.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    loaded = [(name, load_rows(path)) for name, path in SPLITS]
    flip_rate_by_split = compute_flip_rate_by_split()

    outputs = []
    outputs.append(save_heatmap(loaded, MAIN_METRICS, "main", "horizontal", flip_rate_by_split))
    outputs.append(save_heatmap(loaded, APPENDIX_METRICS, "appendix", "vertical", flip_rate_by_split))

    latex_path = BASE_DIR / "refamb_trajectory_quality_heatmaps.tex"
    latex_path.write_text(
        "\n".join(
            [
                r"\begin{figure}[t]",
                r"\centering",
                rf"\includegraphics[width=0.98\linewidth]{{figs/{outputs[0].name}}}",
                r"\caption{Main-text heatmaps for the full Original split. The left subfigure shows the correct subset and the right subfigure shows the incorrect subset. Each subfigure contains RTE and LJ score, with color ranges normalized within each panel to emphasize relative differences.}",
                r"\label{fig:refamb-trajectory-quality-main-heatmap}",
                r"\end{figure}",
                "",
                r"\begin{figure}[t]",
                r"\centering",
                rf"\includegraphics[width=0.98\linewidth]{{figs/{outputs[1].name}}}",
                r"\caption{Appendix heatmaps for the full Original split. The left subfigure shows the correct subset and the right subfigure shows the incorrect subset. Each subfigure contains iteration count, $\Delta F_{avg}$, and $U_{avg}$, with panel-wise normalization.}",
                r"\label{fig:refamb-trajectory-quality-appendix-heatmap}",
                r"\end{figure}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("\n".join(str(path) for path in outputs))
    print(str(latex_path))


if __name__ == "__main__":
    main()
