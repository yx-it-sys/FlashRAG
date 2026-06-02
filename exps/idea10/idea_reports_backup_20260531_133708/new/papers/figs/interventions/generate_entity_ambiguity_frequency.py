#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path("/home/you/FlashRAG/exps/idea10")
RESULT_ROOT = ROOT / "data" / "result"
OUT_ROOT = ROOT / "idea_reports" / "new" / "papers" / "figs" / "interventions"


@dataclass(frozen=True)
class ModelSpec:
    result_dir: Path
    out_dir: Path
    display_name: str


MODEL_SPECS = [
    ModelSpec(
        result_dir=RESULT_ROOT / "RefAmb_original_InternVL3.5-8B",
        out_dir=OUT_ROOT / "internvl3_5_8b",
        display_name="InternVL3.5-8B",
    ),
    ModelSpec(
        result_dir=RESULT_ROOT / "RefAmb_original_Qwen2.5-vl-7B",
        out_dir=OUT_ROOT / "qwen2_5_vl_7b",
        display_name="Qwen2.5-vl-7B",
    ),
    ModelSpec(
        result_dir=RESULT_ROOT / "RefAmb_original_qwen3_vl_32b",
        out_dir=OUT_ROOT / "qwen3_vl_32b",
        display_name="Qwen3-vl-32B",
    ),
    ModelSpec(
        result_dir=RESULT_ROOT / "RefAmb_original_Qwen3-vl-2b",
        out_dir=OUT_ROOT / "qwen3_vl_2b",
        display_name="Qwen3-vl-2B",
    ),
    ModelSpec(
        result_dir=RESULT_ROOT / "RefAmb_original_Qwen3-vl-4B",
        out_dir=OUT_ROOT / "qwen3_vl_4b",
        display_name="Qwen3-vl-4B",
    ),
    ModelSpec(
        result_dir=RESULT_ROOT / "RefAmb_original_Qwen3-vl-8b",
        out_dir=OUT_ROOT / "qwen3_vl_8b",
        display_name="Qwen3-vl-8B",
    ),
]

SOURCE_ORDER = ["oven", "infoseek", "mcsearch", "crag"]
SOURCE_DISPLAY = {
    "oven": "OVEN",
    "infoseek": "InfoSeek",
    "mcsearch": "MCSearch",
    "crag": "CRAG",
}


def iter_label_files(result_dir: Path) -> list[Path]:
    files = []
    for path in result_dir.rglob("omnisearch_trajectories.entity_ambiguity_labeled.jsonl"):
        rel = path.relative_to(result_dir).as_posix()
        if "first_round_oracle_rewrite" in rel or "disturb_rewrite" in rel:
            continue
        files.append(path)
    return sorted(files)


def infer_source(path: Path) -> str:
    text = path.as_posix().lower()
    for source in SOURCE_ORDER:
        if f"_{source}_" in text or f"_{source}-" in text:
            return source
    return "unknown"


def count_labels(label_file: Path) -> dict[str, Any]:
    total = 0
    ambiguous = 0
    not_ambiguous = 0
    other = 0

    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for action in row.get("trajectory", []):
                label = action.get("llm_label")
                if not isinstance(label, dict) or "entity_ambiguous" not in label:
                    continue
                total += 1
                value = label.get("entity_ambiguous")
                if value == "Yes":
                    ambiguous += 1
                elif value == "No":
                    not_ambiguous += 1
                else:
                    other += 1

    rate = ambiguous / total if total else None
    return {
        "total_labels": total,
        "ambiguous_labels": ambiguous,
        "not_ambiguous_labels": not_ambiguous,
        "other_labels": other,
        "ambiguity_rate": rate,
    }


def ensure_dirs(spec: ModelSpec) -> tuple[Path, Path]:
    csv_dir = spec.out_dir / "csv"
    fig_dir = spec.out_dir / "figs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, fig_dir


def format_pct(rate: float | None) -> str:
    if rate is None:
        return "NA"
    return f"{rate * 100:.2f}%"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "source",
        "label_file",
        "total_labels",
        "ambiguous_labels",
        "not_ambiguous_labels",
        "other_labels",
        "ambiguity_rate",
        "ambiguity_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def plot_model_summary(spec: ModelSpec, rows: list[dict[str, Any]], fig_path: Path) -> None:
    data_rows = [row for row in rows if row["source"] != "overall" and row["ambiguity_rate"] is not None]
    if not data_rows:
        return

    labels = [SOURCE_DISPLAY.get(row["source"], row["source"]) for row in data_rows] + ["Overall"]
    values = [float(row["ambiguity_rate"]) for row in data_rows]
    overall = next(row for row in rows if row["source"] == "Overall")
    values.append(float(overall["ambiguity_rate"]))

    colors = ["#2B7DBA"] * len(data_rows) + ["#6DBB4C"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Entity ambiguity frequency")
    ax.set_title(spec.display_name)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, rate in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(rate + 0.025, 0.985),
            f"{rate*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    combined_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"models": {}}

    for spec in MODEL_SPECS:
        csv_dir, fig_dir = ensure_dirs(spec)
        label_files = iter_label_files(spec.result_dir)

        model_rows: list[dict[str, Any]] = []
        source_groups: dict[str, list[Path]] = defaultdict(list)
        for label_file in label_files:
            source_groups[infer_source(label_file)].append(label_file)

        for source in SOURCE_ORDER:
            for label_file in sorted(source_groups.get(source, [])):
                stats = count_labels(label_file)
                row = {
                    "model": spec.display_name,
                    "source": SOURCE_DISPLAY[source],
                    "label_file": str(label_file),
                    **stats,
                    "ambiguity_percent": format_pct(stats["ambiguity_rate"]),
                }
                model_rows.append(row)

        if model_rows:
            total_labels = sum(row["total_labels"] for row in model_rows)
            ambiguous_labels = sum(row["ambiguous_labels"] for row in model_rows)
            not_ambiguous_labels = sum(row["not_ambiguous_labels"] for row in model_rows)
            other_labels = sum(row["other_labels"] for row in model_rows)
            overall_rate = ambiguous_labels / total_labels if total_labels else None
            overall_row = {
                "model": spec.display_name,
                "source": "Overall",
                "label_file": "aggregate",
                "total_labels": total_labels,
                "ambiguous_labels": ambiguous_labels,
                "not_ambiguous_labels": not_ambiguous_labels,
                "other_labels": other_labels,
                "ambiguity_rate": overall_rate,
                "ambiguity_percent": format_pct(overall_rate),
            }
            model_rows.append(overall_row)
        else:
            overall_row = {
                "model": spec.display_name,
                "source": "Overall",
                "label_file": "aggregate",
                "total_labels": 0,
                "ambiguous_labels": 0,
                "not_ambiguous_labels": 0,
                "other_labels": 0,
                "ambiguity_rate": None,
                "ambiguity_percent": "NA",
            }
            model_rows.append(overall_row)

        write_csv(csv_dir / "entity_ambiguity_frequency.csv", model_rows)
        write_json(
            csv_dir / "entity_ambiguity_frequency_summary.json",
            {
                "model": spec.display_name,
                "result_dir": str(spec.result_dir),
                "label_file_count": len(label_files),
                "source_file_count": len(model_rows) - 1,
                "overall": overall_row,
                "sources": model_rows[:-1],
                "missing_label_files": len(label_files) == 0,
            },
        )

        if label_files:
            plot_model_summary(spec, model_rows, fig_dir / "entity_ambiguity_frequency")

        combined_rows.extend(model_rows if len(model_rows) == 1 else model_rows[:-1])
        report["models"][spec.display_name] = {
            "label_file_count": len(label_files),
            "overall_ambiguity_rate": overall_row["ambiguity_rate"],
            "overall_ambiguity_percent": overall_row["ambiguity_percent"],
            "result_dir": str(spec.result_dir),
            "output_dir": str(spec.out_dir),
            "label_files": [str(path) for path in label_files],
        }

    write_csv(OUT_ROOT / "entity_ambiguity_frequency_all_models.csv", combined_rows)
    write_json(OUT_ROOT / "entity_ambiguity_frequency_all_models_summary.json", report)

    report_md = OUT_ROOT / "entity_ambiguity_frequency_report.md"
    lines = ["# Entity Ambiguity Frequency Report", ""]
    lines.append("| Model | Label files | Ambiguous / Total | Frequency |")
    lines.append("|---|---:|---:|---:|")
    for model_name, info in report["models"].items():
        rate = info["overall_ambiguity_percent"]
        if info["label_file_count"] == 0:
            ratio = "NA"
        else:
            csv_path = Path(info["output_dir"]) / "csv" / "entity_ambiguity_frequency_summary.json"
            summary = json.loads(csv_path.read_text(encoding="utf-8"))
            overall = summary["overall"]
            ratio = f'{overall["ambiguous_labels"]} / {overall["total_labels"]}'
        lines.append(f"| {model_name} | {info['label_file_count']} | {ratio} | {rate} |")
    lines.extend(
        [
            "",
            "Notes:",
            "- `Qwen3-vl-2B` has no `omnisearch_trajectories.entity_ambiguity_labeled.jsonl` files in this checkout, so its frequency cannot be computed from the requested labels.",
            "- Frequencies are computed as `#(entity_ambiguous == Yes) / #(all labeled steps)` across the base four source runs for each model.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
