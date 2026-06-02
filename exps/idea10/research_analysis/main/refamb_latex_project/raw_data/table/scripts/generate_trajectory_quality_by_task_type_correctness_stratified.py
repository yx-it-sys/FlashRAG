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

DEFAULT_OUTPUT_CSV = (
    Path("/home/you/FlashRAG/exps/idea10/idea_reports/papers/raw_data")
    / "trajectory_quality_by_task_type_stratified_by_correctness.csv"
)
DEFAULT_OUTPUT_JSON = (
    Path("/home/you/FlashRAG/exps/idea10/idea_reports/papers/raw_data")
    / "trajectory_quality_by_task_type_stratified_by_correctness_summary.json"
)

TASK_ORDER = [
    "Entity Recognition",
    "Single-hop Attribute Query",
    "Multi-hop",
    "Comparison",
    "Subproblem Aggregation",
]
METRIC_ORDER = ["rte", "iter", "delta_f_avg", "u_avg", "lj_score"]
DISTURB_SUBDIR = "first_round_disturb_rewrite_static_prefix_whole"
ORACLE_SUBDIR = "first_round_oracle_rewrite"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected a JSON list in {path}")


def load_first_step_ambiguity(label_row: dict) -> str | None:
    for step in label_row.get("trajectory", []):
        if not isinstance(step, dict):
            continue
        if step.get("action") != "search":
            continue
        llm_label = step.get("llm_label") or {}
        label = llm_label.get("entity_ambiguous")
        if label in {"Yes", "No"}:
            return label
    return None


def resolve_metric_path(stage_dir: Path) -> Path:
    updated = stage_dir / "trajectory_quality_eval_whole_delta_F_updated" / "trajectory_quality_samples.jsonl"
    fallback = stage_dir / "trajectory_quality_eval_whole_delta_F" / "trajectory_quality_samples.jsonl"
    if updated.exists():
        return updated
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing trajectory_quality_samples.jsonl under {stage_dir}")


def load_stage_records(stage_dir: Path) -> list[dict]:
    metric_rows = load_json(stage_dir / "gpt_acc_score.json")
    metric_map = {row["id"]: row for row in metric_rows if row.get("id")}
    disturb_metric_rows = load_json(stage_dir / DISTURB_SUBDIR / "gpt_acc_score.json")
    disturb_metric_map = {row["id"]: row for row in disturb_metric_rows if row.get("id")}
    oracle_metric_rows = load_json(stage_dir / ORACLE_SUBDIR / "gpt_acc_score.json")
    oracle_metric_map = {row["id"]: row for row in oracle_metric_rows if row.get("id")}

    label_path = stage_dir / "label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.jsonl"
    label_rows = load_jsonl(label_path)
    label_map = {row["id"]: row for row in label_rows if row.get("id")}

    tq_rows = load_jsonl(resolve_metric_path(stage_dir))
    tq_map = {row["id"]: row for row in tq_rows if row.get("id")}

    records: list[dict] = []
    for sample_id, metric_row in metric_map.items():
        label_row = label_map.get(sample_id)
        tq_row = tq_map.get(sample_id)
        if label_row is None or tq_row is None:
            continue

        gpt_acc = metric_row.get("output", {}).get("metric_score", {}).get("gpt_acc")
        if not isinstance(gpt_acc, (int, float)):
            continue

        first_label = load_first_step_ambiguity(label_row)
        if first_label is None:
            continue

        steps = tq_row.get("steps") or []
        utilities = [
            float(step.get("utility"))
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("utility"), (int, float))
        ]
        num_iterations = float(tq_row.get("num_iterations", 0) or 0)
        total_delta_f = float(tq_row.get("total_delta_f", 0.0) or 0.0)
        delta_f_avg = (total_delta_f / num_iterations) if num_iterations > 0 else None
        u_avg = mean(utilities) if utilities else None

        records.append(
            {
                "id": sample_id,
                "task_type": metric_row.get("task_type", ""),
                "correctness": "Correct" if float(gpt_acc) == 1.0 else "Incorrect",
                "ambiguity": "Ambiguous" if first_label == "Yes" else "Non-ambiguous",
                "rte": float(tq_row.get("trajectory_quality_score", 0.0) or 0.0),
                "iter": num_iterations,
                "delta_f_avg": delta_f_avg,
                "u_avg": u_avg,
                "lj_score": float(gpt_acc),
                "disturb_flip_ratio": (
                    None
                    if sample_id not in disturb_metric_map
                    else (
                        1.0
                        if (
                            (float(gpt_acc) == 0.0 and float(disturb_metric_map[sample_id]["output"]["metric_score"]["gpt_acc"]) == 1.0)
                            or (
                                float(gpt_acc) == 1.0
                                and float(disturb_metric_map[sample_id]["output"]["metric_score"]["gpt_acc"]) == 0.0
                            )
                        )
                        else 0.0
                    )
                ),
                "oracle_flip_ratio": (
                    None
                    if sample_id not in oracle_metric_map
                    else (
                        1.0
                        if (
                            (float(gpt_acc) == 0.0 and float(oracle_metric_map[sample_id]["output"]["metric_score"]["gpt_acc"]) == 1.0)
                            or (
                                float(gpt_acc) == 1.0
                                and float(oracle_metric_map[sample_id]["output"]["metric_score"]["gpt_acc"]) == 0.0
                            )
                        )
                        else 0.0
                    )
                ),
            }
        )
    return records


def mean_or_blank(values: list[float | None]) -> str:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    return "" if not filtered else f"{mean(filtered):.3f}"


def build_rows(records: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["task_type"], record["correctness"], record["ambiguity"])].append(record)

    rows: list[dict] = []
    summary: dict = {"tasks": {}, "macro_average": {}}

    for task in TASK_ORDER:
        summary["tasks"][task] = {}
        for correctness in ["Correct", "Incorrect"]:
            summary["tasks"][task][correctness] = {}
            for ambiguity in ["Ambiguous", "Non-ambiguous"]:
                items = grouped.get((task, correctness, ambiguity), [])
                summary["tasks"][task][correctness][ambiguity] = {
                    "n": len(items),
                    "rte": mean_or_blank([it["rte"] for it in items]),
                    "iter": mean_or_blank([it["iter"] for it in items]),
                    "delta_f_avg": mean_or_blank([it["delta_f_avg"] for it in items]),
                    "u_avg": mean_or_blank([it["u_avg"] for it in items]),
                    "lj_score": mean_or_blank([it["lj_score"] for it in items]),
                    "disturb_flip_ratio": mean_or_blank([it["disturb_flip_ratio"] for it in items]),
                    "oracle_flip_ratio": mean_or_blank([it["oracle_flip_ratio"] for it in items]),
                }
                rows.append(
                    {
                        "task_type": task,
                        "correctness": correctness,
                        "ambiguity": ambiguity,
                        "n": len(items),
                        "rte": mean_or_blank([it["rte"] for it in items]),
                        "iter": mean_or_blank([it["iter"] for it in items]),
                        "delta_f_avg": mean_or_blank([it["delta_f_avg"] for it in items]),
                        "u_avg": mean_or_blank([it["u_avg"] for it in items]),
                        "lj_score": mean_or_blank([it["lj_score"] for it in items]),
                        "disturb_flip_ratio": mean_or_blank([it["disturb_flip_ratio"] for it in items]),
                        "oracle_flip_ratio": mean_or_blank([it["oracle_flip_ratio"] for it in items]),
                    }
                )

    for correctness in ["Correct", "Incorrect"]:
        summary["macro_average"][correctness] = {}
        for ambiguity in ["Ambiguous", "Non-ambiguous"]:
            task_level_rows = []
            for task in TASK_ORDER:
                item = summary["tasks"][task][correctness][ambiguity]
                if item["n"] > 0:
                    task_level_rows.append(item)

            summary["macro_average"][correctness][ambiguity] = {
                "tasks_covered": len(task_level_rows),
                "n": sum(item["n"] for item in task_level_rows),
                "rte": mean_or_blank([float(item["rte"]) for item in task_level_rows if item["rte"] != ""]),
                "iter": mean_or_blank([float(item["iter"]) for item in task_level_rows if item["iter"] != ""]),
                "delta_f_avg": mean_or_blank(
                    [float(item["delta_f_avg"]) for item in task_level_rows if item["delta_f_avg"] != ""]
                ),
                "u_avg": mean_or_blank([float(item["u_avg"]) for item in task_level_rows if item["u_avg"] != ""]),
                "lj_score": mean_or_blank([float(item["lj_score"]) for item in task_level_rows if item["lj_score"] != ""]),
                "disturb_flip_ratio": mean_or_blank(
                    [float(item["disturb_flip_ratio"]) for item in task_level_rows if item["disturb_flip_ratio"] != ""]
                ),
                "oracle_flip_ratio": mean_or_blank(
                    [float(item["oracle_flip_ratio"]) for item in task_level_rows if item["oracle_flip_ratio"] != ""]
                ),
            }
            rows.append(
                {
                    "task_type": "Macro-average",
                    "correctness": correctness,
                    "ambiguity": ambiguity,
                    "n": sum(item["n"] for item in task_level_rows),
                    "rte": summary["macro_average"][correctness][ambiguity]["rte"],
                    "iter": summary["macro_average"][correctness][ambiguity]["iter"],
                    "delta_f_avg": summary["macro_average"][correctness][ambiguity]["delta_f_avg"],
                    "u_avg": summary["macro_average"][correctness][ambiguity]["u_avg"],
                    "lj_score": summary["macro_average"][correctness][ambiguity]["lj_score"],
                    "disturb_flip_ratio": summary["macro_average"][correctness][ambiguity]["disturb_flip_ratio"],
                    "oracle_flip_ratio": summary["macro_average"][correctness][ambiguity]["oracle_flip_ratio"],
                }
            )

    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_type",
        "correctness",
        "ambiguity",
        "n",
        "rte",
        "iter",
        "delta_f_avg",
        "u_avg",
        "lj_score",
        "disturb_flip_ratio",
        "oracle_flip_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    all_records: list[dict] = []
    stage_counts: dict[str, int] = {}
    for stage_name in STAGE_DIRS:
        stage_dir = RESULT_ROOT / stage_name
        stage_records = load_stage_records(stage_dir)
        all_records.extend(stage_records)
        stage_counts[stage_name] = len(stage_records)

    rows, summary = build_rows(all_records)

    payload = {
        "result_root": str(RESULT_ROOT),
        "stage_counts": stage_counts,
        "total_records": len(all_records),
        "task_order": TASK_ORDER,
        "correctness_levels": ["Correct", "Incorrect"],
        "ambiguity_levels": ["Ambiguous", "Non-ambiguous"],
        "summary": summary,
    }

    write_csv(DEFAULT_OUTPUT_CSV, rows)
    write_json(DEFAULT_OUTPUT_JSON, payload)

    print(f"Wrote CSV: {DEFAULT_OUTPUT_CSV}")
    print(f"Wrote JSON: {DEFAULT_OUTPUT_JSON}")
    print(f"Total records: {len(all_records)}")


if __name__ == "__main__":
    main()
