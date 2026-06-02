#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

TASK_ORDER = [
    "Entity Recognition",
    "Single-hop Attribute Query",
    "Multi-hop",
    "Comparison",
    "Subproblem Aggregation",
]
METRIC_ORDER = ["RTE", "DeltaF", "Util", "LJ", "Itr"]
SOURCE_TOKENS = ["crag", "infoseek", "mcsearch", "oven"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert tab_main_baselines_datasets.csv to task-type aggregated CSV using per-item results."
    )
    p.add_argument(
        "--datasets-csv",
        default="/home/you/FlashRAG/exps/idea10/idea_reports/papers/table/tab_main_baselines_datasets.csv",
    )
    p.add_argument(
        "--task-type-csv",
        default="/home/you/FlashRAG/exps/idea10/idea_reports/papers/table/tab_main_baselines_task_type.csv",
    )
    p.add_argument(
        "--task-subset",
        default="/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/task_balanced_analysis_subset.jsonl",
    )
    p.add_argument(
        "--mapping-json",
        default="",
        help="Optional JSON mapping Method->result_root. If omitted, use built-in defaults.",
    )
    return p.parse_args()


def default_mapping() -> Dict[str, str]:
    root = "/home/you/FlashRAG/exps/idea10/data/result"
    return {
        "Qwen2.5-VL-7B#1": f"{root}/RefAmb_original_Qwen2.5-vl-7B",
        "InternVL3.5#1": f"{root}/RefAmb_original_InternVL3.5-8B",
        "Qwen2.5-VL-7B#2": f"{root}/RefAmb_baseline_prompt_v1_Qwen2.5-vl-7B",
        "ATL-CI#1": f"{root}/RefAmb_ours_Qwen2.5-vl-7B/main",
        "-w/o ROI-grounded retrieval core#1": f"{root}/RefAmb_ours_Qwen2.5-vl-7B/ablation/w.o.ROI",
        "-w/o loosening controls#1": f"{root}/RefAmb_ours_Qwen2.5-vl-7B/ablation/w.o.loosening_controls",
    }


def load_id2task(path: str) -> Dict[str, str]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            out[o["id"]] = o.get("task_type", "")
    return out


def pick_stage_dirs(result_root: str) -> List[str]:
    picked = []
    if not os.path.isdir(result_root):
        return picked
    for tok in SOURCE_TOKENS:
        cands = glob.glob(os.path.join(result_root, f"*{tok}*stage"))
        if not cands:
            continue
        cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        picked.append(cands[0])
    return picked


def load_per_item_metrics(result_root: str) -> Dict[str, Dict[str, float]]:
    per_item: Dict[str, Dict[str, float]] = {}
    for stage in pick_stage_dirs(result_root):
        inter = os.path.join(stage, "intermediate_data.json")
        tqs_updated = os.path.join(
            stage,
            "trajectory_quality_eval_whole_delta_F_updated",
            "trajectory_quality_samples.jsonl",
        )
        tqs_fallback = os.path.join(
            stage,
            "trajectory_quality_eval_whole_delta_F",
            "trajectory_quality_samples.jsonl",
        )
        tqs = tqs_updated if os.path.exists(tqs_updated) else tqs_fallback
        if not (os.path.exists(inter) and os.path.exists(tqs)):
            continue

        lj_map: Dict[str, float] = {}
        try:
            arr = json.load(open(inter, "r", encoding="utf-8"))
            for it in arr:
                iid = it.get("id")
                ms = (it.get("output") or {}).get("metric_score") or {}
                lj = ms.get("llm", ms.get("gpt_acc", None))
                if iid and isinstance(lj, (int, float)):
                    lj_map[iid] = float(lj)
        except Exception:
            pass

        with open(tqs, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                iid = o.get("id")
                if not iid:
                    continue
                n = float(o.get("num_iterations", 0) or 0)
                td = float(o.get("total_delta_f", 0.0) or 0.0)
                steps = o.get("steps") or []
                utils = [s.get("utility") for s in steps if isinstance(s.get("utility"), (int, float))]
                util = (sum(utils) / len(utils)) if utils else None
                per_item[iid] = {
                    "RTE": float(o.get("trajectory_quality_score", 0.0) or 0.0),
                    "DeltaF": (td / n) if n > 0 else None,
                    "Util": util,
                    "Itr": n,
                    "LJ": lj_map.get(iid, None),
                }
    return per_item


def aggregate_by_task(per_item: Dict[str, Dict[str, float]], id2task: Dict[str, str]) -> Dict[str, Dict[str, Optional[float]]]:
    buckets = {t: {m: [] for m in METRIC_ORDER} for t in TASK_ORDER}
    for iid, m in per_item.items():
        t = id2task.get(iid)
        if t not in buckets:
            continue
        for k in METRIC_ORDER:
            v = m.get(k)
            if isinstance(v, (int, float)):
                buckets[t][k].append(float(v))

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for t in TASK_ORDER:
        out[t] = {}
        for k in METRIC_ORDER:
            arr = buckets[t][k]
            out[t][k] = (sum(arr) / len(arr)) if arr else None

    # Overall = macro average over 5 task groups
    out["Overall"] = {}
    for k in METRIC_ORDER:
        vals = [out[t][k] for t in TASK_ORDER if isinstance(out[t][k], (int, float))]
        out["Overall"][k] = (sum(vals) / len(vals)) if vals else None
    return out


def fmt(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.3f}"


def main() -> None:
    args = parse_args()
    id2task = load_id2task(args.task_subset)

    rows = list(csv.DictReader(open(args.datasets_csv, "r", encoding="utf-8")))

    mapping = default_mapping()
    if args.mapping_json:
        mapping = json.load(open(args.mapping_json, "r", encoding="utf-8"))

    header = ["Method"]
    for t in TASK_ORDER:
        prefix = t.replace(" ", "").replace("-", "")
        for m in METRIC_ORDER:
            header.append(f"{prefix}_{m}")
    for m in METRIC_ORDER:
        header.append(f"Overall_{m}")

    method_seen = defaultdict(int)
    out_rows = []
    for r in rows:
        method = (r.get("Method") or "").strip()
        if not method:
            continue
        method_seen[method] += 1
        key = f"{method}#{method_seen[method]}"
        result_root = mapping.get(key)

        out = {"Method": method}
        if result_root and os.path.isdir(result_root):
            per_item = load_per_item_metrics(result_root)
            agg = aggregate_by_task(per_item, id2task)
            for t in TASK_ORDER:
                prefix = t.replace(" ", "").replace("-", "")
                for m in METRIC_ORDER:
                    out[f"{prefix}_{m}"] = fmt(agg[t][m])
            for m in METRIC_ORDER:
                out[f"Overall_{m}"] = fmt(agg["Overall"][m])
        else:
            for c in header[1:]:
                out[c] = "--"
        out_rows.append(out)

    with open(args.task_type_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote: {args.task_type_csv}")


if __name__ == "__main__":
    main()
