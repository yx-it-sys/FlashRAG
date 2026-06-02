#!/usr/bin/env python3
"""
Convert between LaTeX tabular rows and CSV.

Supports three modes:
1) latex2csv: parse rows from a LaTeX table block into CSV.
2) csv2latex: generate LaTeX rows from CSV.
3) csv2fulltable: generate a full table environment for task-type CSV.

Usage examples:
  python latex_csv_converter.py latex2csv \
    --input /path/main.tex \
    --output /path/table.csv \
    --label tab:main-baselines-datasets

  python latex_csv_converter.py csv2latex \
    --input /path/table.csv \
    --output /path/table_rows.tex

  python latex_csv_converter.py csv2fulltable \
    --input /path/main_baselines_category_by_task_type.csv \
    --output /path/main_baselines_category_by_task_type_full.tex
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROW_END = "\\\\"


def _extract_table_block(text: str, label: Optional[str]) -> str:
    if label:
        li = text.find(f"\\label{{{label}}}")
        if li < 0:
            raise ValueError(f"Label not found: {label}")
        begin = text.rfind("\\begin{tabular}", 0, li)
        end = text.find("\\end{tabular}", li)
        if begin < 0 or end < 0:
            raise ValueError(f"Could not locate tabular block for label: {label}")
        end += len("\\end{tabular}")
        return text[begin:end]

    begin = text.find("\\begin{tabular}")
    end = text.find("\\end{tabular}")
    if begin < 0 or end < 0:
        raise ValueError("Could not find any \\begin{tabular}...\\end{tabular} block")
    end += len("\\end{tabular}")
    return text[begin:end]


def _clean_method_cell(cell: str) -> str:
    cell = cell.strip()
    cell = cell.replace("\\textbf{", "")
    cell = cell.replace("}", "")
    cell = cell.replace("\\quad", "").strip()
    return cell


def latex_to_csv(input_path: Path, output_path: Path, label: Optional[str], keep_latex: bool) -> None:
    text = input_path.read_text(encoding="utf-8")
    block = _extract_table_block(text, label)

    rows: List[List[str]] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%"):
            continue
        if any(k in line for k in ["\\toprule", "\\midrule", "\\bottomrule", "\\cmidrule", "\\multicolumn"]):
            continue
        if "&" not in line:
            continue
        if not line.endswith("\\\\"):
            continue

        cells = [c.strip() for c in line[:-2].split("&")]
        if not keep_latex and cells:
            cells[0] = _clean_method_cell(cells[0])
        rows.append(cells)

    if not rows:
        raise ValueError("No data rows parsed from LaTeX")

    width = max(len(r) for r in rows)
    has_header = False
    if rows and all(c.strip() for c in rows[0]) and not re.search(r"\d", "".join(rows[0])):
        has_header = True

    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if has_header:
            for r in rows:
                w.writerow(r)
        else:
            headers = ["col_" + str(i + 1) for i in range(width)]
            w.writerow(headers)
            for r in rows:
                padded = r + [""] * (width - len(r))
                w.writerow(padded)


def csv_to_latex(input_path: Path, output_path: Path, include_header: bool, row_prefix: str) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty")

    out_lines: List[str] = []
    start_idx = 0
    if include_header:
        hdr = rows[0]
        out_lines.append(row_prefix + " & ".join(hdr) + r" \\")
        start_idx = 1

    for r in rows[start_idx:]:
        out_lines.append(row_prefix + " & ".join(r) + r" \\")

    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


TASK_GROUPS: List[Tuple[str, str]] = [
    ("Entity Recognition", "EntityRecognition"),
    ("Single-hop Attribute Query", "SinglehopAttributeQuery"),
    ("Multi-hop", "Multihop"),
    ("Comparison", "Comparison"),
    ("Subproblem Aggregation", "SubproblemAggregation"),
    ("Overall", "Overall"),
]


def _to_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s or s == "--":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _format_metric(metric: str, val: Optional[float]) -> str:
    if val is None:
        return "--"
    if metric == "LJ":
        return f"{val * 100:.2f}"
    return f"{val:.2f}"


def _rank_sets(rows: List[Dict[str, str]], cols: List[str]) -> Tuple[Dict[str, set], Dict[str, set]]:
    best: Dict[str, set] = {c: set() for c in cols}
    second: Dict[str, set] = {c: set() for c in cols}
    for c in cols:
        vals = []
        for r in rows:
            v = _to_float(r.get(c, ""))
            if v is not None:
                vals.append(v)
        uniq = sorted(set(vals), reverse=True)
        if not uniq:
            continue
        best[c].add(uniq[0])
        if len(uniq) > 1:
            second[c].add(uniq[1])
    return best, second


def _style_value(v: str, raw: Optional[float], best_vals: set, second_vals: set) -> str:
    if raw is None:
        return v
    if raw in best_vals:
        return f"\\textcolor{{red}}{{{v}}}"
    if raw in second_vals:
        return f"\\underline{{{v}}}"
    return v


def _method_display_name(name: str) -> str:
    if name == "ATL-CI":
        return "\\textbf{ATL-CI}"
    if name == "w/o tightening":
        return "\\quad -w/o Entity Tightening"
    if name == "w/o rebuild_messages":
        return "\\quad -w/o Exploration Activity"
    if name == "w/o roi":
        return "\\quad -w/o ROI"
    return name


def csv_to_fulltable(input_path: Path, output_path: Path, caption: str, label: str) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV is empty")

    metric_cols: List[str] = []
    for _, p in TASK_GROUPS:
        for m in ["DeltaF", "Util", "LJ"]:
            metric_cols.append(f"{p}_{m}")
    best_vals, second_vals = _rank_sets(rows, metric_cols)

    out: List[str] = []
    out.append("\\begin{table}[htbp]")
    out.append("\\centering")
    out.append("\\scriptsize")
    out.append("\\setlength{\\tabcolsep}{3pt}")
    out.append(f"\\caption{{{caption}}}")
    out.append("\\resizebox{\\linewidth}{!}{%")
    out.append("\\begin{tabular}{l|ccc|ccc|ccc|ccc|ccc|ccc}")
    out.append("\\toprule")
    out.append("\\multirow{2}{*}{\\textbf{Method}} &")
    out.append("\\multicolumn{3}{c|}{\\textbf{Entity Recognition}} &")
    out.append("\\multicolumn{3}{c|}{\\textbf{Single-hop Attribute Query}} &")
    out.append("\\multicolumn{3}{c|}{\\textbf{Multi-hop}} &")
    out.append("\\multicolumn{3}{c|}{\\textbf{Comparison}} &")
    out.append("\\multicolumn{3}{c|}{\\textbf{Subproblem Aggregation}} &")
    out.append("\\multicolumn{3}{c}{\\textbf{Overall}}  " + ROW_END)
    out.append("\\cmidrule(lr){2-4}")
    out.append("\\cmidrule(lr){5-7}")
    out.append("\\cmidrule(lr){8-10}")
    out.append("\\cmidrule(lr){11-13}")
    out.append("\\cmidrule(lr){14-16}")
    out.append("\\cmidrule(lr){17-19}")
    out.append(" & \\textbf{$\\Delta F$} & \\textbf{Util.} & \\textbf{LJ}")
    out.append(" & \\textbf{$\\Delta F$} & \\textbf{Util.} & \\textbf{LJ}")
    out.append(" & \\textbf{$\\Delta F$} & \\textbf{Util.} & \\textbf{LJ}")
    out.append(" & \\textbf{$\\Delta F$} & \\textbf{Util.} & \\textbf{LJ}")
    out.append(" & \\textbf{$\\Delta F$} & \\textbf{Util.} & \\textbf{LJ}")
    out.append(" & \\textbf{$\\Delta F$} & \\textbf{Util.} & \\textbf{LJ} " + ROW_END)
    out.append("\\midrule")

    section_titles = {
        "Qwen2.5-VL-7B#1": "ReAct-style RAG",
        "Qwen2.5-VL-7B#2": "Prompt Optimization of ReAct",
        "Self-Ask + Search#1": "Ambiguity / Retrieval Control Baselines",
    }

    seen: Dict[str, int] = {}
    for r in rows:
        method = (r.get("Method") or "").strip()
        if not method:
            continue
        seen[method] = seen.get(method, 0) + 1
        key = f"{method}#{seen[method]}"
        if key in section_titles:
            out.append(f"\\multicolumn{{19}}{{c}}{{\\textit{{\\textbf{{{section_titles[key]}}}}}}} " + ROW_END)
            out.append("\\midrule")

        cells = [_method_display_name(method)]
        for _, p in TASK_GROUPS:
            for m in ["DeltaF", "Util", "LJ"]:
                col = f"{p}_{m}"
                raw = _to_float(r.get(col, ""))
                disp = _format_metric(m, raw)
                disp = _style_value(disp, raw, best_vals[col], second_vals[col])
                cells.append(disp)
        out.append(" & ".join(cells) + r"\\")

    out.append("\\bottomrule")
    out.append("\\end{tabular}%")
    out.append("}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{table}")

    output_path.write_text("\n".join(out) + "\n", encoding="utf-8")




def csv_to_rte_latex(input_path: Path, output_path: Path, caption: str, label: str) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty")

    header = reader.fieldnames or []
    rte_cols = [c for c in header if c.endswith("_RTE")]
    lj_cols = [c for c in header if c.endswith("_LJ")]
    if not rte_cols:
        raise ValueError("No *_RTE columns found in CSV")

    # Match domain prefixes; if LJ missing for a domain, render as --.
    domains = [c[:-4] for c in rte_cols]
    col_pairs = [(d + "_RTE", d + "_LJ") for d in domains]

    align = "l|" + "|".join(["cc"] * len(domains))

    rank_cols = [c for c in rte_cols + [d + "_LJ" for d in domains] if c in header]
    best_vals, second_vals = _rank_sets(rows, rank_cols)

    out: List[str] = []
    out.append("\\begin{table}[htbp]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\setlength{\\tabcolsep}{3pt}")
    out.append(f"\\caption{{{caption}}}")
    out.append("\\resizebox{\\linewidth}{!}{%")
    out.append(f"\\begin{{tabular}}{{{align}}}")
    out.append("\\toprule")

    # Header row 1
    h1 = ["\\multirow{2}{*}{\\textbf{Method}}"]
    for d in domains:
        h1.append(f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{d}}}}}" if d != domains[-1] else f"\\multicolumn{{2}}{{c}}{{\\textbf{{{d}}}}}")
    out.append(" & ".join(h1) + " " + ROW_END)

    # cmidrules
    left = 2
    for i in range(len(domains)):
        right = left + 1
        out.append(f"\\cmidrule(lr){{{left}-{right}}}")
        left += 2

    # Header row 2
    h2 = [""]
    for _ in domains:
        h2 += ["\\textbf{RTE}", "\\textbf{LJ}"]
    out.append(" & ".join(h2) + " " + ROW_END)
    out.append("\\midrule")

    section_titles = {
        "Qwen2.5-VL-7B#1": "ReAct-style RAG",
        "Qwen2.5-VL-7B#2": "Prompt Optimization of ReAct",
        "Self-Ask + Search#1": "Ambiguity / Retrieval Control Baselines",
    }

    seen: Dict[str, int] = {}
    ncol = 1 + 2 * len(domains)
    for r in rows:
        method = (r.get("Method") or "").strip()
        if not method:
            continue

        seen[method] = seen.get(method, 0) + 1
        key = f"{method}#{seen[method]}"
        if key in section_titles:
            out.append(f"\\multicolumn{{{ncol}}}{{c}}{{\\textit{{\\textbf{{{section_titles[key]}}}}}}} " + ROW_END)
            out.append("\\midrule")

        cells = [_method_display_name(method)]
        for rte_col, lj_col in col_pairs:
            raw_rte = _to_float(r.get(rte_col, ""))
            disp_rte = "--" if raw_rte is None else f"{raw_rte:.2f}"
            if rte_col in best_vals:
                disp_rte = _style_value(disp_rte, raw_rte, best_vals[rte_col], second_vals[rte_col])

            raw_lj = _to_float(r.get(lj_col, "")) if lj_col in header else None
            # Keep existing LJ scale from CSV (no *100 transform)
            disp_lj = "--" if raw_lj is None else f"{raw_lj:.2f}"
            if lj_col in best_vals:
                disp_lj = _style_value(disp_lj, raw_lj, best_vals[lj_col], second_vals[lj_col])

            cells += [disp_rte, disp_lj]

        out.append(" & ".join(cells) + " " + ROW_END)

    out.append("\\bottomrule")
    out.append("\\end{tabular}%")
    out.append("}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{table}")

    output_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert LaTeX table rows and CSV both ways.")
    sub = p.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("latex2csv", help="Parse LaTeX tabular rows into CSV")
    p1.add_argument("--input", required=True, type=Path, help="Input .tex file")
    p1.add_argument("--output", required=True, type=Path, help="Output .csv file")
    p1.add_argument("--label", default=None, help="Optional LaTeX label to locate target table")
    p1.add_argument("--keep-latex", action="store_true", help="Keep raw LaTeX in first column")

    p2 = sub.add_parser("csv2latex", help="Generate LaTeX rows from CSV")
    p2.add_argument("--input", required=True, type=Path, help="Input .csv file")
    p2.add_argument("--output", required=True, type=Path, help="Output .tex file")
    p2.add_argument("--include-header", action="store_true", help="Also output CSV header as first LaTeX row")
    p2.add_argument("--row-prefix", default="", help="Optional prefix per row, e.g. '% ' for commented rows")

    p3 = sub.add_parser("csv2fulltable", help="Generate a full LaTeX table from task-type CSV")
    p3.add_argument("--input", required=True, type=Path, help="Input .csv file")
    p3.add_argument("--output", required=True, type=Path, help="Output .tex file")
    p3.add_argument(
        "--caption",
        default="Main baselines grouped by inference style across four datasets. Metrics are placeholders to be filled after runs.",
        help="Table caption",
    )
    p3.add_argument("--label", default="tab:main-baselines-datasets", help="Table label")

    p4 = sub.add_parser("csv2rte", help="Generate full LaTeX table with Method + RTE columns only")
    p4.add_argument("--input", required=True, type=Path, help="Input .csv file")
    p4.add_argument("--output", required=True, type=Path, help="Output .tex file")
    p4.add_argument("--caption", default="RTE-only comparison.", help="Table caption")
    p4.add_argument("--label", default="tab:rte-only", help="Table label")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "latex2csv":
        latex_to_csv(args.input, args.output, args.label, args.keep_latex)
    elif args.mode == "csv2latex":
        csv_to_latex(args.input, args.output, args.include_header, args.row_prefix)
    elif args.mode == "csv2fulltable":
        csv_to_fulltable(args.input, args.output, args.caption, args.label)
    elif args.mode == "csv2rte":
        csv_to_rte_latex(args.input, args.output, args.caption, args.label)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
