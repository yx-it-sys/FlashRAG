#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib import colors


RESULT_ROOT_DEFAULT = Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_qwen3_vl_32b")
OUT_DIR_DEFAULT = Path("/home/you/FlashRAG/exps/idea10/idea_reports/new/papers/figs/interventions/qwen3_vl_32b")
PREFIX_DEFAULT = "refamb_qwen3_vl_32b"

TASK_ORDER = [
    "Entity Recognition",
    "Single-hop Attribute Query",
    "Multi-hop",
    "Comparison",
    "Subproblem Aggregation",
]
TASK_SHORT = {
    "Entity Recognition": "E.R.",
    "Single-hop Attribute Query": "S.A.Q.",
    "Multi-hop": "M.H.",
    "Comparison": "Comp.",
    "Subproblem Aggregation": "S.A.",
}
SETTINGS = ["Original", "Disturb", "Oracle"]
SETTING_STYLE = {
    "Original": {"color": "#1f77b4", "marker": "o"},
    "Disturb": {"color": "#ff7f0e", "marker": "s"},
    "Oracle": {"color": "#2ca02c", "marker": "^"},
}
N_BINS = 30
BIN_GAMMA = 3.0


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def setting_subdir(setting: str) -> str:
    return {
        "Disturb": "first_round_disturb_rewrite_static_prefix_whole",
        "Oracle": "first_round_oracle_rewrite",
    }[setting]


def resolve_tqs_path(stage_dir: Path, setting: str) -> Path:
    if setting == "Original":
        candidates = [
            stage_dir / "trajectory_quality_eval_whole_delta_F_updated" / "trajectory_quality_samples.jsonl",
            stage_dir / "trajectory_quality_eval_whole_delta_F" / "trajectory_quality_samples.jsonl",
        ]
    else:
        subdir = setting_subdir(setting)
        candidates = [
            stage_dir / subdir / "trajectory_quality_eval_whole_delta_F_updated" / "trajectory_quality_samples.jsonl",
            stage_dir / subdir / "trajectory_quality_eval_whole_delta_F" / "trajectory_quality_samples.jsonl",
        ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Missing trajectory_quality_samples.jsonl for {setting} under {stage_dir}")


def metric_from_lj(row: dict) -> float:
    metric_score = row.get("output", {}).get("metric_score", {})
    for key in ["llm", "gpt_acc"]:
        val = metric_score.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    raise ValueError("No usable LJ field found")


def metric_from_gpt_acc(row: dict) -> float:
    metric_score = row.get("output", {}).get("metric_score", {})
    for key in ["gpt_acc", "llm"]:
        val = metric_score.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    raise ValueError("No gpt_acc/llm field found")


def mean_or_blank(values: list[float | None]) -> str:
    filtered = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return "--" if not filtered else f"{mean(filtered):.3f}"


def build_stage_records(stage_dir: Path) -> list[dict]:
    original_metrics = load_json(stage_dir / "intermediate_data.json")
    original_metric_map = {row["id"]: row for row in original_metrics if row.get("id")}

    disturb_metrics = load_json(stage_dir / setting_subdir("Disturb") / "intermediate_data.json")
    disturb_metric_map = {row["id"]: row for row in disturb_metrics if row.get("id")}

    oracle_metrics = load_json(stage_dir / setting_subdir("Oracle") / "intermediate_data.json")
    oracle_metric_map = {row["id"]: row for row in oracle_metrics if row.get("id")}

    original_tqs_map = {row["id"]: row for row in load_jsonl(resolve_tqs_path(stage_dir, "Original")) if row.get("id")}
    disturb_tqs_map = {row["id"]: row for row in load_jsonl(resolve_tqs_path(stage_dir, "Disturb")) if row.get("id")}
    oracle_tqs_map = {row["id"]: row for row in load_jsonl(resolve_tqs_path(stage_dir, "Oracle")) if row.get("id")}

    records: list[dict] = []
    for sample_id, original_row in original_metric_map.items():
        task_type = original_row.get("task_type", "")
        if task_type not in TASK_ORDER:
            continue
        original_gpt_acc = metric_from_gpt_acc(original_row)
        split_name = "Correct" if abs(original_gpt_acc - 1.0) < 1e-12 else "Incorrect"

        for setting, metric_map, tqs_map in [
            ("Original", original_metric_map, original_tqs_map),
            ("Disturb", disturb_metric_map, disturb_tqs_map),
            ("Oracle", oracle_metric_map, oracle_tqs_map),
        ]:
            metric_row = metric_map.get(sample_id)
            tqs_row = tqs_map.get(sample_id)
            if metric_row is None or tqs_row is None:
                continue

            steps = tqs_row.get("steps") or []
            utilities = [
                float(step.get("utility"))
                for step in steps
                if isinstance(step, dict) and isinstance(step.get("utility"), (int, float))
            ]
            num_iterations = float(tqs_row.get("num_iterations", 0) or 0)
            total_delta_f = float(tqs_row.get("total_delta_f", 0.0) or 0.0)
            delta_f_avg = (total_delta_f / num_iterations) if num_iterations > 0 else None
            u_avg = mean(utilities) if utilities else None
            lj_score = metric_from_lj(metric_row)
            target_gpt_acc = metric_from_gpt_acc(metric_row)

            records.append(
                {
                    "id": sample_id,
                    "task_type": task_type,
                    "correctness": split_name,
                    "setting": setting,
                    "rte": float(tqs_row.get("trajectory_quality_score", 0.0) or 0.0),
                    "iter": num_iterations,
                    "delta_f_avg": delta_f_avg,
                    "u_avg": u_avg,
                    "lj_score": lj_score,
                    "gpt_acc": target_gpt_acc,
                    "flip_ratio": None
                    if setting == "Original"
                    else (
                        1.0
                        if (
                            (split_name == "Correct" and target_gpt_acc == 0.0)
                            or (split_name == "Incorrect" and target_gpt_acc == 1.0)
                        )
                        else 0.0
                    ),
                }
            )
    return records


def build_all_records(result_root: Path) -> list[dict]:
    stage_dirs = sorted([p for p in result_root.iterdir() if p.is_dir() and p.name.endswith("_stage")])
    records: list[dict] = []
    for stage_dir in stage_dirs:
        records.extend(build_stage_records(stage_dir))
    return records


def aggregate_radar(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task_type"], record["setting"])].append(record)

    rows: list[dict] = []
    for task in TASK_ORDER:
        for setting in SETTINGS:
            items = grouped.get((task, setting), [])
            rows.append(
                {
                    "task_type": task,
                    "setting": setting,
                    "rte": mean_or_blank([it["rte"] for it in items]),
                    "mean_tqs": mean_or_blank([it["rte"] for it in items]),
                    "lj_score": mean_or_blank([it["lj_score"] for it in items]),
                    "llm": mean_or_blank([it["lj_score"] for it in items]),
                }
            )
    return rows


def aggregate_heatmaps(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task_type"], record["correctness"], record["setting"])].append(record)

    outputs: dict[str, list[dict]] = {"Correct": [], "Incorrect": []}
    for correctness in ["Correct", "Incorrect"]:
        for task in TASK_ORDER:
            task_row = {"task_type": task}
            for setting in SETTINGS:
                items = grouped.get((task, correctness, setting), [])
                task_row[f"{setting}_rte"] = mean_or_blank([it["rte"] for it in items])
                task_row[f"{setting}_lj_score"] = mean_or_blank([it["lj_score"] for it in items])
                task_row[f"{setting}_iter"] = mean_or_blank([it["iter"] for it in items])
                task_row[f"{setting}_delta_f_avg"] = mean_or_blank([it["delta_f_avg"] for it in items])
                task_row[f"{setting}_u_avg"] = mean_or_blank([it["u_avg"] for it in items])
                task_row[f"{setting}_flip_ratio"] = "--"
            for setting in ["Disturb", "Oracle"]:
                items = grouped.get((task, correctness, setting), [])
                task_row[f"{setting}_flip_ratio"] = mean_or_blank([it["flip_ratio"] for it in items])
            outputs[correctness].append(task_row)
    return outputs


def save_raw_data_csvs(records: list[dict]) -> tuple[Path, Path, Path]:
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    radar_rows = aggregate_radar(records)
    correct_rows = aggregate_heatmaps(records)["Correct"]
    incorrect_rows = aggregate_heatmaps(records)["Incorrect"]

    radar_csv = CSV_DIR / "trajectory_quality_by_task_type.csv"
    correct_csv = CSV_DIR / "trajectory_quality_by_task_type_original_correct.csv"
    incorrect_csv = CSV_DIR / "trajectory_quality_by_task_type_original_incorrect.csv"

    write_csv(
        radar_csv,
        radar_rows,
        ["task_type", "setting", "rte", "mean_tqs", "lj_score", "llm"],
    )
    write_csv(
        correct_csv,
        correct_rows,
        [
            "task_type",
            "Original_rte",
            "Original_lj_score",
            "Original_iter",
            "Original_delta_f_avg",
            "Original_u_avg",
            "Original_flip_ratio",
            "Disturb_rte",
            "Disturb_lj_score",
            "Disturb_iter",
            "Disturb_delta_f_avg",
            "Disturb_u_avg",
            "Disturb_flip_ratio",
            "Oracle_rte",
            "Oracle_lj_score",
            "Oracle_iter",
            "Oracle_delta_f_avg",
            "Oracle_u_avg",
            "Oracle_flip_ratio",
        ],
    )
    write_csv(
        incorrect_csv,
        incorrect_rows,
        [
            "task_type",
            "Original_rte",
            "Original_lj_score",
            "Original_iter",
            "Original_delta_f_avg",
            "Original_u_avg",
            "Original_flip_ratio",
            "Disturb_rte",
            "Disturb_lj_score",
            "Disturb_iter",
            "Disturb_delta_f_avg",
            "Disturb_u_avg",
            "Disturb_flip_ratio",
            "Oracle_rte",
            "Oracle_lj_score",
            "Oracle_iter",
            "Oracle_delta_f_avg",
            "Oracle_u_avg",
            "Oracle_flip_ratio",
        ],
    )

    write_json(
        CSV_DIR / "trajectory_quality_by_task_type_original_correct_summary.json",
        {"correctness": "correct", "total_records": sum(1 for r in records if r["correctness"] == "Correct")},
    )
    write_json(
        CSV_DIR / "trajectory_quality_by_task_type_original_incorrect_summary.json",
        {"correctness": "incorrect", "total_records": sum(1 for r in records if r["correctness"] == "Incorrect")},
    )

    return radar_csv, correct_csv, incorrect_csv


def parse_float(value: str | None) -> float:
    if value is None or value == "" or value == "--":
        return math.nan
    return float(value)


def matrix_range(matrix: list[list[float]]) -> tuple[float, float]:
    values = [value for row in matrix for value in row if not math.isnan(value)]
    if not values:
        return 0.0, 1.0
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return vmin, vmin + 1e-6
    return vmin, vmax


def annotate_heatmap(ax, matrix, vmin: float, vmax: float):
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
    boundaries = [vmin + (vmax - vmin) * ((i / N_BINS) ** BIN_GAMMA) for i in range(N_BINS + 1)]
    cmap = plt.colormaps["YlGnBu"].resampled(N_BINS)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.set_yticks(range(len(TASK_ORDER)))
    ax.set_yticklabels(TASK_ORDER if show_ylabel else [])
    ax.tick_params(axis="both", length=0, labelsize=12)
    ax.set_title(metric_name, pad=8)
    annotate_heatmap(ax, matrix, vmin, vmax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(len(xlabels) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(TASK_ORDER) + 1)], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)


def draw_radar_chart(csv_path: Path) -> tuple[Path, Path]:
    rows = load_csv(csv_path)
    by_task_setting = {(row["task_type"], row["setting"]): row for row in rows}

    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), subplot_kw={"projection": "polar"}, constrained_layout=True)
    angles = [2 * math.pi * i / len(TASK_ORDER) for i in range(len(TASK_ORDER))]
    angles += angles[:1]

    for ax, (metric_key, metric_title) in zip(axes, [("mean_tqs", "RTE"), ("llm", "LJ Score")]):
        series = {}
        for setting in SETTINGS:
            vals = []
            for task in TASK_ORDER:
                row = by_task_setting.get((task, setting))
                vals.append(parse_float(row[metric_key]) if row is not None else math.nan)
            series[setting] = vals

        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([TASK_SHORT[t] for t in TASK_ORDER], fontsize=9)
        ax.set_title(metric_title, pad=10)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)

        values = [v for vals in series.values() for v in vals if not math.isnan(v)]
        if values:
            vmin, vmax = min(values), max(values)
            pad = 0.15 * (vmax - vmin) if not math.isclose(vmin, vmax) else 0.08
            ax.set_ylim(max(0.0, vmin - pad), vmax + pad)

        for zorder, setting in enumerate(["Original", "Oracle", "Disturb"], start=3):
            style = SETTING_STYLE[setting]
            y = series[setting] + series[setting][:1]
            ax.plot(
                angles,
                y,
                label=setting,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=5.5,
                linestyle="--" if setting == "Disturb" else "-",
                zorder=zorder,
            )
            ax.fill(angles, y, color=style["color"], alpha=0.03, zorder=1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13))

    fig_dir = OUT_DIR / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / f"{PREFIX}_rte_lj_radar.pdf"
    png_path = fig_dir / f"{PREFIX}_rte_lj_radar.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path


def draw_heatmap(csv_paths: list[tuple[str, Path]], basename: str) -> tuple[Path, Path]:
    split_rows = [(split_name, load_csv(csv_path)) for split_name, csv_path in csv_paths]

    plt.rcParams.update(
        {
            "font.size": 16,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    fig = plt.figure(figsize=(21.0, 6.2), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, wspace=0.05)
    if len(split_rows) == 1:
        subfigs = [subfigs]

    for idx, (subfig, (split_name, rows)) in enumerate(zip(subfigs, split_rows)):
        by_task = {row["task_type"]: row for row in rows}
        axes = subfig.subplots(1, 3, squeeze=False)[0]
        for ax_idx, (metric_key, metric_title) in enumerate(
            [
                ("rte", "RTE"),
                ("lj_score", "LJ Score"),
                ("flip_ratio", "Flip Ratio"),
            ]
        ):
            matrix = []
            for task in TASK_ORDER:
                row = by_task[task]
                if metric_key == "flip_ratio":
                    matrix.append([parse_float(row[f"{setting}_{metric_key}"]) for setting in ["Disturb", "Oracle"]])
                else:
                    matrix.append([parse_float(row[f"{setting}_{metric_key}"]) for setting in SETTINGS])
            xlabels = ["Disturb", "Oracle"] if metric_key == "flip_ratio" else SETTINGS
            draw_panel(axes[ax_idx], matrix, metric_title, show_ylabel=(idx == 0 and ax_idx == 0), xlabels=xlabels)
        subfig.text(0.5, -0.08, f"Original-{split_name}", ha="center", va="top", fontsize=18)

    fig_dir = OUT_DIR / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / f"{PREFIX}_{basename}.pdf"
    png_path = fig_dir / f"{PREFIX}_{basename}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Generate trajectory quality radar and heatmap figures.")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--prefix", type=str, default=PREFIX_DEFAULT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    global OUT_DIR, PREFIX, CSV_DIR
    OUT_DIR = args.out_dir
    PREFIX = args.prefix
    CSV_DIR = OUT_DIR / "csv"
    records = build_all_records(args.result_root)
    radar_csv, correct_csv, incorrect_csv = save_raw_data_csvs(records)
    outputs = [
        *draw_radar_chart(radar_csv),
        *draw_heatmap([
            ("correct", correct_csv),
            ("incorrect", incorrect_csv),
        ], "original_main_heatmap"),
    ]
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
