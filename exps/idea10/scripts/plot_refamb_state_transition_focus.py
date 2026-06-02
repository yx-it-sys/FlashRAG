#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B")
OUT_PDF = Path("/home/you/FlashRAG/exps/idea10/scripts/refamb_state_transition_focus.pdf")
OUT_PNG = Path("/home/you/FlashRAG/exps/idea10/scripts/refamb_state_transition_focus.png")

STAGES = [
    ("mcsearch", "RefAmb_2026_05_01_11_26_refamb_mcsearch_stage"),
    ("oven", "RefAmb_2026_05_01_13_28_refamb_oven_stage"),
    ("infoseek", "RefAmb_2026_05_02_13_50_refamb_infoseek_stage"),
    ("crag", "RefAmb_2026_05_02_14_22_refamb_crag_stage"),
]

STATE_LABELS = {
    "ok": "OK",
    "missing_final_answer": "Missing Final Answer",
    "generation_error": "Generation Error",
}


def load_status_map(path: Path) -> dict[str, str]:
    status_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            status_map[row["id"]] = row.get("status", "unknown")
    return status_map


def resolve_path(stage_dir: Path, relative_candidates: list[Path]) -> Path:
    for candidate in relative_candidates:
        path = stage_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No matching file found under {stage_dir}")


def collect_transitions(compare: str) -> tuple[dict[str, Counter], Counter, int]:
    per_source: dict[str, Counter] = {}
    overall: Counter = Counter()

    for source, stage in STAGES:
        stage_dir = BASE_DIR / stage
        original_path = resolve_path(
            stage_dir,
            [
                Path("omnisearch_trajectories.jsonl"),
            ],
        )
        if compare == "oracle":
            candidate_path = resolve_path(
                stage_dir,
                [
                    Path("first_round_oracle_rewrite/omnisearch_trajectories.jsonl"),
                ],
            )
        elif compare == "disturb":
            candidate_path = resolve_path(
                stage_dir,
                [
                    Path("first_round_disturb_rewrite_static_prefix/omnisearch_trajectories.jsonl"),
                ],
            )
        else:
            raise ValueError(compare)

        original = load_status_map(original_path)
        candidate = load_status_map(candidate_path)
        common_ids = original.keys() & candidate.keys()
        counter = Counter((original[item_id], candidate[item_id]) for item_id in common_ids)
        per_source[source] = counter
        overall.update(counter)

    total = sum(overall.values())
    return per_source, overall, total


def to_rate(numer: int, denom: int) -> float:
    return 0.0 if denom == 0 else numer / denom


def build_focus_rows(compare: str, overall: Counter) -> list[dict]:
    if compare == "oracle":
        total = Counter()
        for (orig, _new), count in overall.items():
            total[orig] += count
        return [
            {
                "label": "OK -> OK",
                "numerator": overall[("ok", "ok")],
                "denominator": total["ok"],
            },
            {
                "label": "MFA -> OK",
                "numerator": overall[("missing_final_answer", "ok")],
                "denominator": total["missing_final_answer"],
            },
            {
                "label": "GE -> OK",
                "numerator": overall[("generation_error", "ok")],
                "denominator": total["generation_error"],
            },
            {
                "label": "GE -> MFA",
                "numerator": overall[("generation_error", "missing_final_answer")],
                "denominator": total["generation_error"],
            },
        ]

    total_ok = sum(count for (orig, _new), count in overall.items() if orig == "ok")
    return [
        {
            "label": "OK -> MFA",
            "numerator": overall[("ok", "missing_final_answer")],
            "denominator": total_ok,
        },
        {
            "label": "OK -> GE",
            "numerator": overall[("ok", "generation_error")],
            "denominator": total_ok,
        },
    ]


def annotate_bars(ax, bars, rows, color="#222222") -> None:
    for bar, row in zip(bars, rows):
        pct = 100.0 * to_rate(row["numerator"], row["denominator"])
        text = f'{row["numerator"]}/{row["denominator"]}\n{pct:.1f}%'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            text,
            ha="center",
            va="bottom",
            fontsize=11,
            color=color,
        )


def style_axes(ax, title: str, ylim: float) -> None:
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_ylim(0.0, ylim)
    ax.set_ylabel("Transition rate", fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_panel(ax, rows, color: str) -> None:
    x = list(range(len(rows)))
    values = [to_rate(row["numerator"], row["denominator"]) for row in rows]
    bars = ax.bar(x, values, width=0.62, color=color, edgecolor="#2A2A2A", linewidth=0.8)
    ax.set_xticks(x, [row["label"] for row in rows], rotation=0)
    annotate_bars(ax, bars, rows)


def main() -> None:
    per_source_oracle, overall_oracle, total_oracle = collect_transitions("oracle")
    per_source_disturb, overall_disturb, total_disturb = collect_transitions("disturb")

    oracle_rows = build_focus_rows("oracle", overall_oracle)
    disturb_rows = build_focus_rows("disturb", overall_disturb)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), constrained_layout=True)

    plot_panel(axes[0], oracle_rows, color="#2F6C8F")
    style_axes(axes[0], "Oracle vs Original", ylim=0.8)
    axes[0].set_xlabel("Focus transitions", fontsize=12)
    axes[0].text(
        0.5,
        -0.24,
        f"N={total_oracle} common items across 4 sources",
        ha="center",
        va="top",
        transform=axes[0].transAxes,
        fontsize=10,
        color="#444444",
    )

    plot_panel(axes[1], disturb_rows, color="#C46E2E")
    style_axes(axes[1], "Disturb vs Original", ylim=0.25)
    axes[1].set_xlabel("Focus transitions", fontsize=12)
    axes[1].text(
        0.5,
        -0.24,
        f"N={total_disturb} common items across 4 sources",
        ha="center",
        va="top",
        transform=axes[1].transAxes,
        fontsize=10,
        color="#444444",
    )

    fig.suptitle(
        "Focused State Transitions on Common Items",
        fontsize=16,
        y=1.04,
    )

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)

    print("Saved:", OUT_PDF)
    print("Saved:", OUT_PNG)
    print("Oracle overall:", dict(overall_oracle))
    print("Disturb overall:", dict(overall_disturb))


if __name__ == "__main__":
    main()
