#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULT_ROOT = Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B")
STAGE_DIRS = [
    "RefAmb_2026_05_01_11_26_refamb_mcsearch_stage",
    "RefAmb_2026_05_01_13_28_refamb_oven_stage",
    "RefAmb_2026_05_02_13_50_refamb_infoseek_stage",
    "RefAmb_2026_05_02_14_22_refamb_crag_stage",
]

SETTING_SPECS = {
    "Original": "",
    "Disturb": "first_round_disturb_rewrite_static_prefix_whole",
    "Oracle": "first_round_oracle_rewrite",
}

TASK_ORDER = [
    "Entity Recognition",
    "Single-hop Attribute Query",
    "Multi-hop",
    "Comparison",
    "Subproblem Aggregation",
]

OUTPUT_DIR = Path("/home/you/FlashRAG/exps/idea10/idea_reports/papers/raw_data")
CSV_TEMPLATE = "trajectory_quality_by_task_type_original_{correctness}.csv"
JSON_TEMPLATE = "trajectory_quality_by_task_type_original_{correctness}_summary.json"
TEX_TEMPLATE = "refamb_trajectory_quality_by_task_type_original_{correctness}.tex"

METRIC_KEYS = ["rte", "iter", "delta_f_avg", "u_avg", "lj_score"]


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


def resolve_tqs_path(stage_dir: Path, setting: str) -> Path:
    rel = SETTING_SPECS[setting]
    if not rel:
        candidates = [
            stage_dir / "trajectory_quality_eval_whole_delta_F_updated" / "trajectory_quality_samples.jsonl",
            stage_dir / "trajectory_quality_eval_whole_delta_F" / "trajectory_quality_samples.jsonl",
        ]
    else:
        candidates = [
            stage_dir / rel / "trajectory_quality_eval_whole_delta_F_updated" / "trajectory_quality_samples.jsonl",
            stage_dir / rel / "trajectory_quality_eval_whole_delta_F" / "trajectory_quality_samples.jsonl",
        ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Missing trajectory_quality_samples.jsonl for {setting} under {stage_dir}")


def load_setting_maps(stage_dir: Path, setting: str) -> tuple[dict[str, dict], dict[str, dict]]:
    if setting == "Original":
        metric_rows = load_json(stage_dir / "gpt_acc_score.json")
    else:
        metric_rows = load_json(stage_dir / SETTING_SPECS[setting] / "gpt_acc_score.json")
    metric_map = {row["id"]: row for row in metric_rows if row.get("id")}

    tqs_rows = load_jsonl(resolve_tqs_path(stage_dir, setting))
    tqs_map = {row["id"]: row for row in tqs_rows if row.get("id")}
    return metric_map, tqs_map


def mean_str(values: list[float | None]) -> str:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    return "--" if not filtered else f"{mean(filtered):.3f}"


def build_records_for_correctness(target_correctness: bool) -> tuple[list[dict], dict, int]:
    records: list[dict] = []
    stage_stats = {}
    split_sample_count = 0

    for stage_name in STAGE_DIRS:
        stage_dir = RESULT_ROOT / stage_name
        original_map, original_tqs = load_setting_maps(stage_dir, "Original")
        disturb_map, disturb_tqs = load_setting_maps(stage_dir, "Disturb")
        oracle_map, oracle_tqs = load_setting_maps(stage_dir, "Oracle")

        original_ids = sorted(original_map)
        stage_stats[stage_name] = {
            "original_total": len(original_ids),
            "original_correct": 0,
            "original_incorrect": 0,
            "disturb_available": len(disturb_map),
            "oracle_available": len(oracle_map),
            "disturb_oracle_overlap": len(set(disturb_map) & set(oracle_map)),
            "selected_samples": 0,
        }

        for sample_id in original_ids:
            original_item = original_map[sample_id]
            original_gpt_acc = float(original_item["output"]["metric_score"]["gpt_acc"])
            is_correct = abs(original_gpt_acc - 1.0) < 1e-12
            stage_stats[stage_name]["original_correct" if is_correct else "original_incorrect"] += 1
            if is_correct != target_correctness:
                continue

            stage_stats[stage_name]["selected_samples"] += 1
            split_sample_count += 1

            task_type = original_item.get("task_type", "")
            if task_type not in TASK_ORDER:
                continue

            for setting, metric_map, tqs_map in [
                ("Original", original_map, original_tqs),
                ("Disturb", disturb_map, disturb_tqs),
                ("Oracle", oracle_map, oracle_tqs),
            ]:
                if sample_id not in metric_map or sample_id not in tqs_map:
                    continue
                metric_item = metric_map[sample_id]
                tqs_item = tqs_map[sample_id]
                steps = tqs_item.get("steps") or []
                utilities = [
                    float(step.get("utility"))
                    for step in steps
                    if isinstance(step, dict) and isinstance(step.get("utility"), (int, float))
                ]
                num_iterations = float(tqs_item.get("num_iterations", 0) or 0)
                total_delta_f = float(tqs_item.get("total_delta_f", 0.0) or 0.0)
                delta_f_avg = (total_delta_f / num_iterations) if num_iterations > 0 else None
                u_avg = mean(utilities) if utilities else None
                records.append(
                    {
                        "stage": stage_name,
                        "id": sample_id,
                        "task_type": task_type,
                        "setting": setting,
                        "rte": float(tqs_item.get("trajectory_quality_score", 0.0) or 0.0),
                        "iter": num_iterations,
                        "delta_f_avg": delta_f_avg,
                        "u_avg": u_avg,
                        "lj_score": float(metric_item["output"]["metric_score"]["gpt_acc"]),
                    }
                )

    return records, stage_stats, split_sample_count


def aggregate(records: list[dict], target_correctness: bool) -> tuple[list[dict], dict]:
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["task_type"], r["setting"])].append(r)

    task_types_present = []
    for task in TASK_ORDER:
        if any((task, setting) in grouped for setting in ["Original", "Disturb", "Oracle"]):
            task_types_present.append(task)

    rows: list[dict] = []
    per_task_summary: dict[str, dict] = {}
    for task in task_types_present:
        per_task_summary[task] = {}
        original_by_id = {it["id"]: it for it in grouped.get((task, "Original"), [])}
        for setting in ["Original", "Disturb", "Oracle"]:
            items = grouped.get((task, setting), [])
            flip_ratio = "--"
            if setting in {"Disturb", "Oracle"}:
                flip_flags: list[float] = []
                for item in items:
                    original_item = original_by_id.get(item["id"])
                    if original_item is None:
                        continue
                    if target_correctness:
                        flip_flags.append(1.0 if float(item["lj_score"]) == 0.0 else 0.0)
                    else:
                        flip_flags.append(1.0 if float(item["lj_score"]) == 1.0 else 0.0)
                flip_ratio = mean_str(flip_flags)
            per_task_summary[task][setting] = {
                "n": len(items),
                "rte": mean_str([it["rte"] for it in items]),
                "iter": mean_str([it["iter"] for it in items]),
                "delta_f_avg": mean_str([it["delta_f_avg"] for it in items]),
                "u_avg": mean_str([it["u_avg"] for it in items]),
                "lj_score": mean_str([it["lj_score"] for it in items]),
                "flip_ratio": flip_ratio,
            }
        rows.append(
            {
                "task_type": task,
                "Original_rte": per_task_summary[task]["Original"]["rte"],
                "Original_iter": per_task_summary[task]["Original"]["iter"],
                "Original_delta_f_avg": per_task_summary[task]["Original"]["delta_f_avg"],
                "Original_u_avg": per_task_summary[task]["Original"]["u_avg"],
                "Original_lj_score": per_task_summary[task]["Original"]["lj_score"],
                "Original_flip_ratio": "--",
                "Disturb_rte": per_task_summary[task]["Disturb"]["rte"],
                "Disturb_iter": per_task_summary[task]["Disturb"]["iter"],
                "Disturb_delta_f_avg": per_task_summary[task]["Disturb"]["delta_f_avg"],
                "Disturb_u_avg": per_task_summary[task]["Disturb"]["u_avg"],
                "Disturb_lj_score": per_task_summary[task]["Disturb"]["lj_score"],
                "Disturb_flip_ratio": per_task_summary[task]["Disturb"]["flip_ratio"],
                "Oracle_rte": per_task_summary[task]["Oracle"]["rte"],
                "Oracle_iter": per_task_summary[task]["Oracle"]["iter"],
                "Oracle_delta_f_avg": per_task_summary[task]["Oracle"]["delta_f_avg"],
                "Oracle_u_avg": per_task_summary[task]["Oracle"]["u_avg"],
                "Oracle_lj_score": per_task_summary[task]["Oracle"]["lj_score"],
                "Oracle_flip_ratio": per_task_summary[task]["Oracle"]["flip_ratio"],
            }
        )

    macro = {}
    for setting in ["Original", "Disturb", "Oracle"]:
        vals = [per_task_summary[task][setting] for task in task_types_present]
        macro[setting] = {
            "rte": mean_str([float(v["rte"]) for v in vals if v["rte"] != "--"]),
            "iter": mean_str([float(v["iter"]) for v in vals if v["iter"] != "--"]),
            "delta_f_avg": mean_str([float(v["delta_f_avg"]) for v in vals if v["delta_f_avg"] != "--"]),
            "u_avg": mean_str([float(v["u_avg"]) for v in vals if v["u_avg"] != "--"]),
            "lj_score": mean_str([float(v["lj_score"]) for v in vals if v["lj_score"] != "--"]),
            "flip_ratio": mean_str([float(v["flip_ratio"]) for v in vals if v["flip_ratio"] != "--"]),
        }

    for setting in ["Original", "Disturb", "Oracle"]:
        rows.append(
            {
                "task_type": "Macro-average",
                f"{setting}_rte": macro[setting]["rte"],
                f"{setting}_iter": macro[setting]["iter"],
                f"{setting}_delta_f_avg": macro[setting]["delta_f_avg"],
                f"{setting}_u_avg": macro[setting]["u_avg"],
                f"{setting}_lj_score": macro[setting]["lj_score"],
            }
        )

    # Normalize macro rows into a single row-like dict for CSV/JSON convenience.
    macro_row = {
        "task_type": "Macro-average",
        "Original_rte": macro["Original"]["rte"],
        "Original_iter": macro["Original"]["iter"],
        "Original_delta_f_avg": macro["Original"]["delta_f_avg"],
        "Original_u_avg": macro["Original"]["u_avg"],
        "Original_lj_score": macro["Original"]["lj_score"],
        "Original_flip_ratio": "--",
        "Disturb_rte": macro["Disturb"]["rte"],
        "Disturb_iter": macro["Disturb"]["iter"],
        "Disturb_delta_f_avg": macro["Disturb"]["delta_f_avg"],
        "Disturb_u_avg": macro["Disturb"]["u_avg"],
        "Disturb_lj_score": macro["Disturb"]["lj_score"],
        "Disturb_flip_ratio": macro["Disturb"]["flip_ratio"],
        "Oracle_rte": macro["Oracle"]["rte"],
        "Oracle_iter": macro["Oracle"]["iter"],
        "Oracle_delta_f_avg": macro["Oracle"]["delta_f_avg"],
        "Oracle_u_avg": macro["Oracle"]["u_avg"],
        "Oracle_lj_score": macro["Oracle"]["lj_score"],
        "Oracle_flip_ratio": macro["Oracle"]["flip_ratio"],
    }
    rows[-3:] = [macro_row]  # replace temporary macro rows with a single row

    summary = {
        "task_types_present": task_types_present,
        "per_task_summary": per_task_summary,
        "macro_average": macro,
        "records": len(records),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "task_type",
        "Original_rte",
        "Original_iter",
        "Original_delta_f_avg",
        "Original_u_avg",
        "Original_lj_score",
        "Original_flip_ratio",
        "Disturb_rte",
        "Disturb_iter",
        "Disturb_delta_f_avg",
        "Disturb_u_avg",
        "Disturb_lj_score",
        "Disturb_flip_ratio",
        "Oracle_rte",
        "Oracle_iter",
        "Oracle_delta_f_avg",
        "Oracle_u_avg",
        "Oracle_lj_score",
        "Oracle_flip_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def format_cell(value: str) -> str:
    return value if value != "--" else "--"


def build_tex(correctness_label: str, rows: list[dict], summary: dict) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Trajectory quality metrics on the full Original pool where the Original setting is "
        + correctness_label
        + r". The pool is formed by merging the four RefAmb sources and then splitting by the Original setting's GPT accuracy. Disturb and Oracle are averaged over the samples available in each setting; they are not intersected with each other.}"
    )
    lines.append(r"\begin{tabular}{lllllll}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Task Type} & \textbf{Setting} & \textbf{RTE} & \textbf{\#Iter.} & \textbf{$\Delta F_{\text{avg}}$} & \textbf{$U_{avg}$} & \textbf{LJ score} \\")
    lines.append(r"\hline")

    task_types_present = summary["task_types_present"]
    row_by_task = {row["task_type"]: row for row in rows}
    for task in task_types_present:
        row = row_by_task[task]
        # Original
        lines.append(
            f"\\multirow{{3}}{{*}}{{{task}}} & Original & {format_cell(row['Original_rte'])} & {format_cell(row['Original_iter'])} & {format_cell(row['Original_delta_f_avg'])} & {format_cell(row['Original_u_avg'])} & {format_cell(row['Original_lj_score'])} \\\\"
        )
        lines.append(
            f" & Disturb & {format_cell(row['Disturb_rte'])} & {format_cell(row['Disturb_iter'])} & {format_cell(row['Disturb_delta_f_avg'])} & {format_cell(row['Disturb_u_avg'])} & {format_cell(row['Disturb_lj_score'])} \\\\"
        )
        lines.append(
            f" & Oracle & {format_cell(row['Oracle_rte'])} & {format_cell(row['Oracle_iter'])} & {format_cell(row['Oracle_delta_f_avg'])} & {format_cell(row['Oracle_u_avg'])} & {format_cell(row['Oracle_lj_score'])} \\\\"
        )
        lines.append(r"\hline")

    macro = row_by_task["Macro-average"]
    lines.append(
        f"\\multirow{{3}}{{*}}{{Macro-average}} & Original & {format_cell(macro['Original_rte'])} & {format_cell(macro['Original_iter'])} & {format_cell(macro['Original_delta_f_avg'])} & {format_cell(macro['Original_u_avg'])} & {format_cell(macro['Original_lj_score'])} \\\\"
    )
    lines.append(
        f" & Disturb & {format_cell(macro['Disturb_rte'])} & {format_cell(macro['Disturb_iter'])} & {format_cell(macro['Disturb_delta_f_avg'])} & {format_cell(macro['Disturb_u_avg'])} & {format_cell(macro['Disturb_lj_score'])} \\\\"
    )
    lines.append(
        f" & Oracle & {format_cell(macro['Oracle_rte'])} & {format_cell(macro['Oracle_iter'])} & {format_cell(macro['Oracle_delta_f_avg'])} & {format_cell(macro['Oracle_u_avg'])} & {format_cell(macro['Oracle_lj_score'])} \\\\"
    )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    all_stage_common = {}
    outputs = {}

    for correctness_label, target in [("correct", True), ("incorrect", False)]:
        records, stage_stats, split_sample_count = build_records_for_correctness(target)
        rows, summary = aggregate(records, target)
        csv_path = OUTPUT_DIR / CSV_TEMPLATE.format(correctness=correctness_label)
        json_path = OUTPUT_DIR / JSON_TEMPLATE.format(correctness=correctness_label)
        tex_path = OUTPUT_DIR / TEX_TEMPLATE.format(correctness=correctness_label)
        write_csv(csv_path, rows)
        json_path.write_text(
            json.dumps(
                {
                    "correctness": correctness_label,
                    "stage_stats": stage_stats,
                    "split_sample_count": split_sample_count,
                    "summary": summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        tex_path.write_text(build_tex("correct" if target else "incorrect", rows, summary), encoding="utf-8")
        outputs[correctness_label] = {
            "csv": str(csv_path),
            "json": str(json_path),
            "tex": str(tex_path),
            "split_sample_count": split_sample_count,
            "records": len(records),
            "tasks": summary["task_types_present"],
        }

    manifest = OUTPUT_DIR / "trajectory_quality_by_task_type_original_correctness_manifest.json"
    manifest.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
