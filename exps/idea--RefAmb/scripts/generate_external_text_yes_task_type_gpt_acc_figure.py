#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B')
OUTPUT_DIR = Path('/home/you/FlashRAG/exps/idea10/idea_reports/refamb_analysis_figs')
SUBSET_PATH = Path('/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/task_balanced_analysis_subset.jsonl')
INTERMEDIATE_GLOB = 'RefAmb_*_refamb_*_stage/intermediate_data.json'
TASK_ORDER = [
    'Single-hop Attribute Query',
    'Entity Recognition',
    'Subproblem Aggregation',
    'Multi-hop',
    'Comparison',
]
TASK_LABELS = {
    'Single-hop Attribute Query': 'Single-hop\nAttribute',
    'Entity Recognition': 'Entity\nRecognition',
    'Subproblem Aggregation': 'Subproblem\nAggregation',
    'Multi-hop': 'Multi-hop',
    'Comparison': 'Comparison',
}


def style() -> None:
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 16,
            'axes.titlesize': 18,
            'axes.labelsize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'axes.spines.top': True,
            'axes.spines.right': True,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
        }
    )


def load_task_types() -> dict[str, str]:
    task_by_id = {}
    with SUBSET_PATH.open('r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            task_by_id[row['id']] = row['task_type']
    return task_by_id


def build_rows():
    task_by_id = load_task_types()
    grouped = defaultdict(list)

    for path in sorted(ROOT.glob(INTERMEDIATE_GLOB)):
        if 'first_round_oracle_rewrite' in str(path):
            continue
        with path.open('r', encoding='utf-8') as f:
            rows = json.load(f)
        for row in rows:
            sample_id = row['id']
            task_type = task_by_id.get(sample_id)
            gpt_acc = ((row.get('output') or {}).get('metric_score') or {}).get('gpt_acc')
            if (
                task_type is None
                or gpt_acc is None
                or row.get('external_evidence_dependency') != 'Yes'
                or row.get('text_alone_identifies_entity') != 'No'
            ):
                continue
            grouped[task_type].append(float(gpt_acc))

    plot_rows = []
    for task_type in TASK_ORDER:
        values = grouped.get(task_type, [])
        if not values:
            continue
        plot_rows.append(
            {
                'task_type': task_type,
                'label': TASK_LABELS.get(task_type, task_type),
                'mean': float(np.mean(np.array(values, dtype=float))),
                'count': len(values),
            }
        )
    return plot_rows


def get_plot_order(rows):
    mean_by_task = {row['task_type']: row['mean'] for row in rows}
    return sorted(
        TASK_ORDER,
        key=lambda task_type: (task_type not in mean_by_task, mean_by_task.get(task_type, float('inf'))),
    )


def plot(rows):
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    plot_order = get_plot_order(rows)
    row_map = {row['task_type']: row for row in rows}
    x_arr = np.arange(len(plot_order))
    mean_arr = np.array([row_map[task_type]['mean'] for task_type in plot_order])
    color = '#4c78a8'

    ax.set_xticks(x_arr, [TASK_LABELS.get(task_type, task_type) for task_type in plot_order])
    ax.set_ylabel('GPT_ACC')
    ax.grid(axis='y', linestyle='--', alpha=0.22)

    ax.plot(x_arr, mean_arr, 'o', color=color, markersize=8, zorder=3)
    ax.plot(x_arr, mean_arr, color=color, linewidth=1.8, alpha=0.8, zorder=2)

    for x, mean in zip(x_arr, mean_arr):
        ax.text(
            x,
            mean + 0.02,
            f'{mean:.3f}',
            ha='center',
            va='bottom',
            fontsize=13,
            color=color,
        )

    ax.set_ylim(0.0, max(1.0, float(np.max(mean_arr)) + 0.08))
    return fig


def save(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f'{name}.pdf'
    png_path = OUTPUT_DIR / f'{name}.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    style()
    rows = build_rows()
    if not rows:
        raise RuntimeError('No matched samples found.')
    pdf_path, png_path = save(
        plot(rows),
        'task_type_gpt_acc_point_range',
    )
    print(pdf_path)
    print(png_path)
    for row in rows:
        print(f"{row['task_type']}\tcount={row['count']}\tmean={row['mean']:.6f}")


if __name__ == '__main__':
    main()
