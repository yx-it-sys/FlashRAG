#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path("/home/you/FlashRAG/exps/idea10")
INTERVENTIONS_DIR = ROOT / "idea_reports" / "new" / "papers" / "figs" / "interventions"
MECH_DIR = INTERVENTIONS_DIR / "mechanism"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


BAR_MODULE = load_module(INTERVENTIONS_DIR / "generate_trajectory_quality_bar_charts.py")

TASK_ORDER = BAR_MODULE.TASK_ORDER
TASK_DISPLAY_LABELS = BAR_MODULE.TASK_DISPLAY_LABELS
SETTINGS = BAR_MODULE.SETTINGS
DATASET_SPECS = BAR_MODULE.DATASET_SPECS
SETTING_STYLE = BAR_MODULE.SETTING_STYLE


@dataclass(frozen=True)
class MechanismSummary:
    model: str
    setting: str
    task_balanced: dict[str, float]
    per_task: dict[str, dict[str, float]]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_paths(stage_dir: Path, setting: str) -> tuple[Path, Path]:
    if setting == "Original":
        meta = stage_dir / "intermediate_data.json"
        candidates = [
            stage_dir / "trajectory_quality_eval_whole_delta_F_updated" / "trajectory_quality_samples.jsonl",
            stage_dir / "trajectory_quality_eval_whole_delta_F" / "trajectory_quality_samples.jsonl",
        ]
    else:
        subdir = BAR_MODULE.SETTING_SUBDIR[setting]
        meta = stage_dir / subdir / "intermediate_data.json"
        candidates = [
            stage_dir / subdir / "trajectory_quality_eval_whole_delta_F_updated" / "trajectory_quality_samples.jsonl",
            stage_dir / subdir / "trajectory_quality_eval_whole_delta_F" / "trajectory_quality_samples.jsonl",
        ]
    for cand in candidates:
        if cand.exists():
            return meta, cand
    raise FileNotFoundError(f"Missing trajectory_quality_samples.jsonl for {setting} under {stage_dir}")


def task_balanced_mean(task_map: dict[str, dict[str, list[float]]], metric: str) -> float | None:
    vals = []
    for task in TASK_ORDER:
        series = task_map.get(task, {}).get(metric, [])
        if series:
            vals.append(mean(series))
    return None if not vals else mean(vals)


def build_mechanism_summary(result_root: Path, model_label: str) -> MechanismSummary:
    stage_dirs = sorted([p for p in result_root.iterdir() if p.is_dir() and p.name.endswith("_stage")])
    per_setting: dict[str, dict[str, dict[str, list[float]]]] = {}

    for setting in SETTINGS:
        per_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for stage_dir in stage_dirs:
            meta_path, tqs_path = resolve_paths(stage_dir, setting)
            if not meta_path.exists() or not tqs_path.exists():
                continue
            task_map = {row["id"]: row.get("task_type") for row in load_json(meta_path) if row.get("id")}
            for row in load_jsonl(tqs_path):
                task = task_map.get(row["id"])
                if task not in TASK_ORDER:
                    continue
                per_task[task]["tqs"].append(float(row["trajectory_quality_score"]))
                per_task[task]["delta_f"].append(float(row["total_delta_f"]))
                per_task[task]["iters"].append(float(row["num_iterations"]))
                per_task[task]["queries"].extend(
                    [float(len(step.get("query", "").split())) for step in row.get("steps", []) if isinstance(step.get("query"), str)]
                )
                per_task[task]["query_sims"].extend(
                    [float(step["query_similarity_with_previous"]) for step in row.get("steps", []) if isinstance(step.get("query_similarity_with_previous"), (int, float))]
                )
                per_task[task]["step_u"].extend(
                    [float(step["utility"]) for step in row.get("steps", []) if isinstance(step.get("utility"), (int, float))]
                )
                per_task[task]["step_score"].extend(
                    [float(step["step_score"]) for step in row.get("steps", []) if isinstance(step.get("step_score"), (int, float))]
                )
                per_task[task]["step_delta_f"].extend(
                    [float(step["delta_f"]) for step in row.get("steps", []) if isinstance(step.get("delta_f"), (int, float))]
                )
                per_task[task]["inf_steps"].extend(
                    [
                        1.0
                        for step in row.get("steps", [])
                        if isinstance(step.get("utility"), (int, float))
                        and isinstance(step.get("delta_f"), (int, float))
                        and step["utility"] > 0
                        and step["delta_f"] > 0
                    ]
                )
                per_task[task]["step_count"].extend([1.0 for _ in row.get("steps", [])])

        per_setting[setting] = per_task

    task_balanced: dict[str, dict[str, float]] = {}
    per_task_stats: dict[str, dict[str, float]] = {}
    for setting, task_map in per_setting.items():
        task_balanced[setting] = {
            "tqs": task_balanced_mean(task_map, "tqs"),
            "delta_f": task_balanced_mean(task_map, "delta_f"),
            "iters": task_balanced_mean(task_map, "iters"),
            "step_u": task_balanced_mean(task_map, "step_u"),
            "step_score": task_balanced_mean(task_map, "step_score"),
            "step_delta_f": task_balanced_mean(task_map, "step_delta_f"),
            "query_len": task_balanced_mean(task_map, "queries"),
            "query_sim": task_balanced_mean(task_map, "query_sims"),
            "inf_rate": None,
        }
        inf_vals = []
        for task in TASK_ORDER:
            if task_map.get(task, {}).get("step_count"):
                inf_vals.append(sum(task_map[task]["inf_steps"]) / sum(task_map[task]["step_count"]))
        task_balanced[setting]["inf_rate"] = None if not inf_vals else mean(inf_vals)

        for task in TASK_ORDER:
            series = task_map.get(task, {})
            if not series:
                continue
            per_task_stats[f"{setting}:{task}"] = {
                "model": model_label,
                "setting": setting,
                "task": task,
                "tqs": mean(series["tqs"]) if series.get("tqs") else None,
                "delta_f": mean(series["delta_f"]) if series.get("delta_f") else None,
                "iters": mean(series["iters"]) if series.get("iters") else None,
                "step_u": mean(series["step_u"]) if series.get("step_u") else None,
                "step_score": mean(series["step_score"]) if series.get("step_score") else None,
                "step_delta_f": mean(series["step_delta_f"]) if series.get("step_delta_f") else None,
                "query_len": mean(series["queries"]) if series.get("queries") else None,
                "query_sim": mean(series["query_sims"]) if series.get("query_sims") else None,
                "inf_rate": (sum(series["inf_steps"]) / sum(series["step_count"])) if series.get("step_count") else None,
                "n_trajectories": len(series["tqs"]),
                "n_steps": int(sum(series["step_count"])) if series.get("step_count") else 0,
            }

    return MechanismSummary(model=model_label, setting="all", task_balanced=task_balanced, per_task=per_task_stats)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def plot_overview(points: list[dict[str, Any]], out_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )

    models = list(dict.fromkeys(point["model"] for point in points))
    model_colors = {model: color for model, color in zip(models, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])}
    setting_order = ["Original", "Disturb", "Oracle"]
    setting_markers = {"Original": "o", "Disturb": "s", "Oracle": "^"}
    setting_offsets = {"Original": (-0.01, 0.01), "Disturb": (0.01, -0.01), "Oracle": (0.0, 0.0)}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    panels = [
        ("step_u", "tqs", "Mean utility", "Task-balanced TQS"),
        ("delta_f", "tqs", "Mean ΔF", "Task-balanced TQS"),
    ]
    for ax, (x_key, y_key, x_label, y_label) in zip(axes, panels):
        for model in models:
            model_points = [p for p in points if p["model"] == model]
            model_points = sorted(model_points, key=lambda p: setting_order.index(p["setting"]))
            xs = [p[x_key] for p in model_points if p[x_key] is not None and p[y_key] is not None]
            ys = [p[y_key] for p in model_points if p[x_key] is not None and p[y_key] is not None]
            if len(xs) < 2:
                continue
            color = model_colors[model]
            ax.plot(xs, ys, "-", color=color, alpha=0.7, linewidth=1.4)
            for point in model_points:
                if point[x_key] is None or point[y_key] is None:
                    continue
                dx, dy = setting_offsets[point["setting"]]
                ax.scatter(
                    point[x_key],
                    point[y_key],
                    s=58,
                    marker=setting_markers[point["setting"]],
                    color=color,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )
                ax.text(
                    point[x_key] + dx,
                    point[y_key] + dy,
                    point["setting"][0],
                    fontsize=9,
                    color=color,
                    ha="center",
                    va="center",
                )
            orig = model_points[0]
            ax.text(orig[x_key], orig[y_key], f" {model}", fontsize=9, color=color, ha="left", va="bottom")

        ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker=setting_markers[s], color="black", linestyle="None", markersize=7, label=s)
        for s in setting_order
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_task_breakdown(task_rows: list[dict[str, Any]], out_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )
    metrics = [
        ("tqs", "TQS"),
        ("delta_f", "ΔF"),
        ("step_u", "Utility"),
        ("inf_rate", "Informative rate"),
    ]
    tasks = TASK_ORDER
    n_cols = len(metrics)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 4.3), constrained_layout=True, sharey=False)
    if n_cols == 1:
        axes = [axes]

    width = 0.22
    offsets = {"Original": -width, "Disturb": 0.0, "Oracle": width}
    rows_by_setting_task = {(row["setting"], row["task"]): row for row in task_rows}

    for ax, (metric, title) in zip(axes, metrics):
        for setting in SETTINGS:
            vals = [rows_by_setting_task[(setting, task)][metric] for task in tasks if (setting, task) in rows_by_setting_task]
            ax.bar(
                [i + offsets[setting] for i in range(len(tasks))],
                vals,
                width=width,
                color=SETTING_STYLE[setting]["color"],
                label=setting if ax is axes[0] else None,
                edgecolor="white",
                linewidth=0.6,
            )
        ax.set_title(title)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([TASK_DISPLAY_LABELS[t] for t in tasks], rotation=18, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Task mean")
    fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MECH_DIR.mkdir(parents=True, exist_ok=True)

    overview_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for model_name, spec in DATASET_SPECS.items():
        if model_name == "Qwen3-vl-2B":
            continue
        summary = build_mechanism_summary(spec["result_root"], model_name)
        summaries[model_name] = {
            "model": model_name,
            "result_root": str(spec["result_root"]),
            "task_balanced": summary.task_balanced,
        }
        for setting in SETTINGS:
            row = {
                "model": model_name,
                "setting": setting,
                **summary.task_balanced[setting],
            }
            overview_rows.append(row)
        task_rows.extend(summary.per_task.values())

    write_csv(MECH_DIR / "mechanism_overview_task_balanced.csv", overview_rows)
    write_json(MECH_DIR / "mechanism_overview_task_balanced_summary.json", summaries)
    write_csv(MECH_DIR / "mechanism_qwen3_vl_32b_task_breakdown.csv", [row for row in task_rows if row["model"] == "Qwen3-vl-32B"])
    write_json(
        MECH_DIR / "mechanism_qwen3_vl_32b_task_breakdown.json",
        {
            "model": "Qwen3-vl-32B",
            "rows": [row for row in task_rows if row["model"] == "Qwen3-vl-32B"],
        },
    )

    plot_overview(overview_rows, MECH_DIR / "mechanism_overview_scatter")
    plot_task_breakdown([row for row in task_rows if row["model"] == "Qwen3-vl-32B"], MECH_DIR / "mechanism_qwen3_vl_32b_task_breakdown")

    analysis_lines = [
        "# Mechanism Analysis",
        "",
        "This supplement focuses on why rewrite changes trajectory quality, using the same task-balanced aggregation convention as the bar charts.",
        "",
        "## Core Mechanism",
        "",
        "The dominant effect of both `Disturb` and `Oracle` is not a simple increase in fact novelty. Instead, the rewrites usually push the agent into longer trajectories with higher `ΔF` but lower `utility` and lower informative-step density.",
        "",
        "In other words, the intervention often increases the amount of retrieved material while decreasing the fraction of steps that the downstream reasoning actually uses.",
        "",
        "## Backbone-Level Pattern",
        "",
        "| Model | Setting | TQS | ΔF | Iterations | Utility | Informative rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in overview_rows:
        analysis_lines.append(
            f"| {row['model']} | {row['setting']} | {row['tqs']:.4f} | {row['delta_f']:.4f} | {row['iters']:.4f} | {row['step_u']:.4f} | {row['inf_rate']:.4f} |"
        )
    analysis_lines += [
        "",
        "### Key observations",
        "",
        "- `Qwen2.5-vl-7B` is the only backbone where both `Disturb` and `Oracle` improve `TQS` relative to `Original`, and `Oracle` also raises `utility` and the informative-step rate. This is the regime where explicit grounding behaves like a net positive control signal.",
        "- `Qwen3-vl-32B` shows the opposite mechanism: `Oracle` raises `ΔF` and `iterations` the most, but it also produces the lowest `utility` and the lowest informative-step rate among the Qwen3 family. This is the strongest evidence that explicit disambiguation can over-constrain the retrieval policy rather than help it.",
        "- `Qwen3-vl-4B` and `Qwen3-vl-8B` sit in the middle: `Oracle` partially recovers `TQS` on 4B but still leaves utility below the original level on 8B. The rewrite effect is therefore family-dependent, not a one-way monotonic gain.",
        "",
        "## Task-Level Mechanism on Qwen3-vl-32B",
        "",
        "| Task | Setting | TQS | ΔF | Iterations | Utility | Informative rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    q32_task_rows = [row for row in task_rows if row["model"] == "Qwen3-vl-32B"]
    q32_task_rows = sorted(q32_task_rows, key=lambda r: (TASK_ORDER.index(r["task"]), SETTINGS.index(r["setting"])))
    for row in q32_task_rows:
        analysis_lines.append(
            f"| {row['task']} | {row['setting']} | {row['tqs']:.4f} | {row['delta_f']:.4f} | {row['iters']:.4f} | {row['step_u']:.4f} | {row['inf_rate']:.4f} |"
        )
    analysis_lines += [
        "",
        "### Task-level reading",
        "",
        "- `Entity Recognition` is the only task family where `Oracle` clearly lifts `TQS` above `Original`, which suggests that explicit entity anchoring helps when the task is primarily about naming the referent.",
        "- `Multi-hop` and `Comparison` are the failure modes: under `Oracle`, both tasks show a sharp rise in iterations and `ΔF` but a collapse in utility and `TQS`. This is the clearest mechanistic sign that explicit grounding can break the search policy on reasoning-heavy tasks.",
        "- `Subproblem Aggregation` remains high-utility even under rewrite, which explains why the backbone-level averages do not collapse completely. The intervention is therefore selective, not uniformly harmful.",
        "",
        "## Mechanistic Interpretation",
        "",
        "The figure-level interpretation is that referential disambiguation changes the search policy more than it changes the information content of the query. For Qwen3 backbones, the model often keeps retrieving longer chains of evidence, but the utility gate weakens, so more retrieved facts are not translated into proportionally better trajectory quality.",
        "",
        "This is the key mechanism statement to carry into the paper: the controllability gap is not just about ambiguity frequency. It is about whether the backbone treats explicit grounding as a useful anchor or as an over-constraining signal that reduces useful exploration.",
    ]
    (MECH_DIR / "analysis.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
