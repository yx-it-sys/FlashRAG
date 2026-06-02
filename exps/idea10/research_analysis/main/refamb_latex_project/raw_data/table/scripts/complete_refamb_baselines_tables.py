#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


TASK_ORDER = [
    "Entity Recognition",
    "Single-hop Attribute Query",
    "Multi-hop",
    "Comparison",
    "Subproblem Aggregation",
]

DATASET_ORDER = ["crag", "infoseek", "mcsearch", "oven"]
METRIC_ORDER = ["RTE", "DeltaF", "Util", "LJ", "Itr"]
SOURCE_TOKENS = ["crag", "infoseek", "mcsearch", "oven"]

DEFAULT_TASK_TYPE_CSV = "/home/you/FlashRAG/exps/idea10/research_analysis/main/refamb_latex_project/raw_data/main_baselines_category_by_task_type.csv"
DEFAULT_TASK_TYPE_TEX = "/home/you/FlashRAG/exps/idea10/research_analysis/main/refamb_latex_project/raw_data/main_baselines_category_by_task_type.tex"
DEFAULT_DATASETS_CSV = "/home/you/FlashRAG/exps/idea10/research_analysis/main/refamb_latex_project/raw_data/main_baselines_category_by_datasets.csv"
DEFAULT_TASK_SUBSET = "/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/task_balanced_analysis_subset.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fill the RefAmb main baseline CSV tables from a shared data/result tree. "
            "Pass any family root under /data/result (for example RefAmb_original_Qwen2.5-vl-7B)."
        )
    )
    p.add_argument(
        "--result-root",
        default="/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B",
        help="Any result-family root under the shared data/result tree. Its parent directory is used as the scan base.",
    )
    p.add_argument(
        "--results-base",
        default="",
        help="Override the shared results base directory. If omitted, use parent(result-root).",
    )
    p.add_argument(
        "--task-subset",
        default=DEFAULT_TASK_SUBSET,
        help="JSONL file containing the task_type and source annotations used for aggregation.",
    )
    p.add_argument(
        "--task-type-csv",
        default=DEFAULT_TASK_TYPE_CSV,
        help="Output CSV path for the task-type table.",
    )
    p.add_argument(
        "--task-type-tex",
        default=DEFAULT_TASK_TYPE_TEX,
        help="Output LaTeX path for the task-type table.",
    )
    p.add_argument(
        "--datasets-csv",
        default=DEFAULT_DATASETS_CSV,
        help="Output CSV path for the dataset table.",
    )
    return p.parse_args()


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_task_metadata(path: str) -> tuple[Dict[str, str], Dict[str, str]]:
    id2task: Dict[str, str] = {}
    id2source: Dict[str, str] = {}
    for row in load_jsonl(path):
        iid = row.get("id")
        if not iid:
            continue
        id2task[iid] = row.get("task_type", "")
        id2source[iid] = row.get("source", "")
    return id2task, id2source


def pick_stage_dirs(result_root: str) -> List[str]:
    picked: List[str] = []
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


def aggregate_by_group(
    per_item: Dict[str, Dict[str, float]],
    id2group: Dict[str, str],
    group_order: List[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    buckets = {g: {m: [] for m in METRIC_ORDER} for g in group_order}
    for iid, metrics in per_item.items():
        group = id2group.get(iid, "")
        if group not in buckets:
            continue
        for k in METRIC_ORDER:
            v = metrics.get(k)
            if isinstance(v, (int, float)):
                buckets[group][k].append(float(v))

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for g in group_order:
        out[g] = {}
        for k in METRIC_ORDER:
            arr = buckets[g][k]
            out[g][k] = (sum(arr) / len(arr)) if arr else None

    out["Overall"] = {}
    for k in METRIC_ORDER:
        vals = [out[g][k] for g in group_order if isinstance(out[g][k], (int, float))]
        out["Overall"][k] = (sum(vals) / len(vals)) if vals else None
    return out


def fmt(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.3f}"


def as_root(base_root: Path, maybe_rel: str) -> str:
    p = Path(maybe_rel)
    return str(p if p.is_absolute() else base_root / p)


def build_table_rows(
    template_csv: str,
    mapping: Dict[str, str],
    id2group: Dict[str, str],
    group_order: List[str],
    prefix_map: Dict[str, str],
) -> tuple[list[dict], list[str]]:
    with open(template_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []

    if not rows:
        raise ValueError(f"Empty template CSV: {template_csv}")

    metric_cols: List[str] = [c for c in header if c != "Method"]
    seen = defaultdict(int)
    out_rows: list[dict] = []

    for row in rows:
        method = (row.get("Method") or "").strip()
        if not method:
            continue
        seen[method] += 1
        key = f"{method}#{seen[method]}"
        out = {"Method": method}
        root = mapping.get(key, "")
        if root and os.path.isdir(root):
            per_item = load_per_item_metrics(root)
            agg = aggregate_by_group(per_item, id2group, group_order)
            for group in group_order:
                prefix = prefix_map[group]
                for metric in METRIC_ORDER:
                    out[f"{prefix}_{metric}"] = fmt(agg[group][metric])
            for metric in METRIC_ORDER:
                out[f"Overall_{metric}"] = fmt(agg["Overall"][metric])
        else:
            for col in metric_cols:
                out[col] = ""
        out_rows.append(out)

    return out_rows, header


def write_csv(path: str, header: list[str], rows: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _metric_to_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    if not value or value == "--":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _tex_metric(value: str) -> str:
    return "--" if _metric_to_float(value) is None else f"{float(value):.2f}"


def _rank_values(rows: list[dict], cols: list[str]) -> tuple[dict[str, set], dict[str, set]]:
    best: dict[str, set] = {c: set() for c in cols}
    second: dict[str, set] = {c: set() for c in cols}
    for c in cols:
        vals: list[float] = []
        for row in rows:
            v = _metric_to_float(row.get(c, ""))
            if v is not None:
                vals.append(v)
        uniq = sorted(set(vals), reverse=True)
        if not uniq:
            continue
        best[c].add(uniq[0])
        if len(uniq) > 1:
            second[c].add(uniq[1])
    return best, second


def _style_tex_value(raw: str, best_vals: set, second_vals: set) -> str:
    v = _metric_to_float(raw)
    if v is None:
        return "--"
    out = _tex_metric(raw)
    if v in best_vals:
        return f"\\textcolor{{red}}{{{out}}}"
    if v in second_vals:
        return f"\\underline{{{out}}}"
    return out


def render_task_type_tex(rows: list[dict], output_path: str) -> None:
    key_counts: dict[str, int] = defaultdict(int)
    row_map: dict[str, dict] = {}
    for row in rows:
        method = (row.get("Method") or "").strip()
        if not method:
            continue
        key_counts[method] += 1
        row_map[f"{method}#{key_counts[method]}"] = row

    selected_sections = [
        (
            "ReAct-style RAG",
            [
                ("Qwen2.5-VL-7B#1", "Qwen2.5-VL-7B"),
                ("Qwen3-VL-32B-Instruct#1", "Qwen3-VL-32B-Instruct"),
                ("GPT-5.1#1", "GPT-5.1"),
            ],
        ),
        (
            "Reinforcement Learning",
            [
                ("MMSearch-R1#1", "MMSearch-R1"),
            ],
        ),
        (
            "Ours",
            [
                ("IIGCR (GPT-5.1)#1", "\\textbf{IIGCR (GPT-5.1)}"),
                ("IIGCR (Qwen3-vl-32b)#1", "IIGCR (Qwen3-vl-32b)"),
                ("IIGCR (Qwen2.5-vl-7b)#1", "IIGCR (Qwen2.5-vl-7b)"),
                ("w/o tightening#1", "\\quad w/o tightening"),
                ("w/o rebuild_messages#1", "\\quad w/o rebuild_messages"),
                ("w/o roi#1", "\\quad w/o roi"),
            ],
        ),
    ]

    tex_rows: list[tuple[str, str, dict]] = []
    for _, items in selected_sections:
        for key, display in items:
            row = row_map.get(key, {})
            tex_rows.append((key, display, row))

    style_cols = [
        "EntityRecognition_RTE",
        "EntityRecognition_LJ",
        "SinglehopAttributeQuery_RTE",
        "SinglehopAttributeQuery_LJ",
        "Multihop_RTE",
        "Multihop_LJ",
        "Comparison_RTE",
        "Comparison_LJ",
        "SubproblemAggregation_RTE",
        "SubproblemAggregation_LJ",
        "Overall_RTE",
        "Overall_LJ",
    ]
    best_vals, second_vals = _rank_values([r for _, _, r in tex_rows], style_cols)

    out: list[str] = []
    out.append("\\begin{table}[htbp]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\setlength{\\tabcolsep}{3pt}")
    out.append("\\caption{Main baselines grouped by inference style across task types.}")
    out.append("\\resizebox{0.9\\linewidth}{!}{%")
    out.append("\\begin{tabular}{lcccccccccccc}")
    out.append("\\toprule")
    out.append(
        "\\multirow{2}{*}{\\textbf{Method}} & "
        "\\multicolumn{2}{c}{\\textbf{E.R.}} & "
        "\\multicolumn{2}{c}{\\textbf{S.A.Q.}} & "
        "\\multicolumn{2}{c}{\\textbf{M.H.}} & "
        "\\multicolumn{2}{c}{\\textbf{Comp.}} & "
        "\\multicolumn{2}{c}{\\textbf{C.C.Q.}} & "
        "\\multicolumn{2}{c}{\\textbf{Overall}} \\\\"
    )
    out.append("\\cmidrule(lr){2-3}")
    out.append("\\cmidrule(lr){4-5}")
    out.append("\\cmidrule(lr){6-7}")
    out.append("\\cmidrule(lr){8-9}")
    out.append("\\cmidrule(lr){10-11}")
    out.append("\\cmidrule(lr){12-13}")
    out.append(
        " & \\textbf{RTE} & \\textbf{LJ} & "
        "\\textbf{RTE} & \\textbf{LJ} & "
        "\\textbf{RTE} & \\textbf{LJ} & "
        "\\textbf{RTE} & \\textbf{LJ} & "
        "\\textbf{RTE} & \\textbf{LJ} & "
        "\\textbf{RTE} & \\textbf{LJ} \\\\"
    )
    out.append("\\midrule")

    for section_idx, (section_title, items) in enumerate(selected_sections):
        out.append(f"\\multicolumn{{1}}{{l}}{{}} & \\multicolumn{{12}}{{c}}{{\\textit{{\\textbf{{{section_title}}}}}}} \\\\")
        out.append("\\midrule")
        for key, display in items:
            row = row_map.get(key, {})
            cells = [display]
            metric_keys = [
                "EntityRecognition_RTE",
                "EntityRecognition_LJ",
                "SinglehopAttributeQuery_RTE",
                "SinglehopAttributeQuery_LJ",
                "Multihop_RTE",
                "Multihop_LJ",
                "Comparison_RTE",
                "Comparison_LJ",
                "SubproblemAggregation_RTE",
                "SubproblemAggregation_LJ",
                "Overall_RTE",
                "Overall_LJ",
            ]
            for col in metric_keys:
                raw = row.get(col, "")
                cells.append(_style_tex_value(raw, best_vals[col], second_vals[col]))
            out.append(" & ".join(cells) + " \\\\")
        if section_idx != len(selected_sections) - 1:
            out.append("\\midrule")

    out.append("\\bottomrule")
    out.append("\\end{tabular}%")
    out.append("}")
    out.append("\\label{tab:main-baselines-tasktype-rte-lj}")
    out.append("\\end{table}")

    tex_path = Path(output_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_root = Path(args.results_base).expanduser().resolve() if args.results_base else Path(args.result_root).expanduser().resolve().parent

    id2task, id2source = load_task_metadata(args.task_subset)

    task_type_mapping = {
        "Qwen2.5-VL-7B#1": as_root(base_root, "RefAmb_original_Qwen2.5-vl-7B"),
        "InternVL3.5#1": as_root(base_root, "RefAmb_original_InternVL3.5-8B"),
        "Qwen3-VL-4B-Instruct#1": as_root(base_root, "RefAmb_original_Qwen3-vl-4B"),
        "Qwen3-VL-8B-Instruct#1": as_root(base_root, "RefAmb_original_Qwen3-vl-8b"),
        "Qwen3-VL-32B-Instruct#1": as_root(base_root, "RefAmb_original_qwen3_vl_32b"),
        "Qwen2.5-VL-7B#2": as_root(base_root, "RefAmb_baseline_prompt_v1_Qwen2.5-vl-7B"),
        "GPT-5.1#1": as_root(base_root, "RefAmb_original_GPT_5.1"),
        "MMSearch-R1#1": "",
        "IIGCR (GPT-5.1)#1": "",
        "IIGCR (Qwen3-vl-32b)#1": as_root(base_root, "RefAmb_ours_Qwen3-vl-32b"),
        "IIGCR (Qwen2.5-vl-7b)#1": as_root(base_root, "RefAmb_ours_Qwen2.5-vl-7B/failure_threshold_0.05/main"),
        "w/o tightening#1": as_root(base_root, "RefAmb_ours_Qwen2.5-vl-7B/failure_threshold_0.05/ablation/w.o.tightening_controller"),
        "w/o rebuild_messages#1": as_root(base_root, "RefAmb_ours_Qwen2.5-vl-7B/failure_threshold_0.05/ablation/w.o.rebuild_messages"),
        "w/o roi#1": as_root(base_root, "RefAmb_ours_Qwen2.5-vl-7B/failure_threshold_0.05/ablation/w.o.ROI"),
    }

    datasets_mapping = {
        "Qwen2.5-VL-7B#1": as_root(base_root, "RefAmb_original_Qwen2.5-vl-7B"),
        "InternVL3.5-8B#1": as_root(base_root, "RefAmb_original_InternVL3.5-8B"),
        "Qwen3-VL-32B-Instruct#1": as_root(base_root, "RefAmb_original_qwen3_vl_32b"),
        "Qwen2.5-VL-7B#2": as_root(base_root, "RefAmb_baseline_prompt_v1_Qwen2.5-vl-7B"),
        "GPT-5.1#1": as_root(base_root, "RefAmb_original_GPT_5.1"),
        "MMSearch-R1#1": "",
        "IIGCR (GPT-5.1)#1": "",
        "IIGCR (Qwen3-vl-32b)#1": as_root(base_root, "RefAmb_ours_Qwen3-vl-32b"),
        "IIGCR (Qwen2.5-vl-7b)#1": as_root(base_root, "RefAmb_ours_Qwen2.5-vl-7B/failure_threshold_0.05/main"),
        "w/o tightening#1": as_root(base_root, "RefAmb_ours_Qwen2.5-vl-7B/failure_threshold_0.05/ablation/w.o.tightening_controller"),
        "w/o rebuild_messages#1": as_root(base_root, "RefAmb_ours_Qwen2.5-vl-7B/failure_threshold_0.05/ablation/w.o.rebuild_messages"),
        "w/o roi#1": as_root(base_root, "RefAmb_ours_Qwen2.5-vl-7B/failure_threshold_0.05/ablation/w.o.ROI"),
    }

    task_type_rows, task_type_header = build_table_rows(
        args.task_type_csv,
        task_type_mapping,
        id2task,
        TASK_ORDER,
        {t: t.replace(" ", "").replace("-", "") for t in TASK_ORDER},
    )
    dataset_rows, dataset_header = build_table_rows(
        args.datasets_csv,
        datasets_mapping,
        id2source,
        DATASET_ORDER,
        {
            "crag": "CRAG",
            "infoseek": "InfoSeek",
            "mcsearch": "MCSearch",
            "oven": "OVEN",
        },
    )

    write_csv(args.task_type_csv, task_type_header, task_type_rows)
    write_csv(args.datasets_csv, dataset_header, dataset_rows)
    render_task_type_tex(task_type_rows, args.task_type_tex)

    print(f"Wrote: {args.task_type_csv}")
    print(f"Wrote: {args.datasets_csv}")
    print(f"Wrote: {args.task_type_tex}")


if __name__ == "__main__":
    main()
