#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B')
OUT_DIR = Path('/home/you/FlashRAG/exps/idea10/idea_reports/refamb_analysis_figs')
OUT_NAME = 'combined_binary_and_tasktype_gptacc'
INTERMEDIATE_GLOB = 'RefAmb_*_refamb_*_stage/intermediate_data.json'
LABEL_GLOB = 'RefAmb_*_refamb_*_stage/label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.jsonl'
REWRITE_LABEL_GLOB = 'RefAmb_*_refamb_*_stage/label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.rewrite.jsonl'
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
ENTITY_GROUPS = ['Yes', 'No']
ENTITY_STYLES = {
    'Yes': {'label': 'Entity Ambiguous = Yes', 'color': '#A8D0E6'},
    'No': {'label': 'Entity Ambiguous = No', 'color': '#F4A261'},
}


def load_module(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_style() -> None:
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


def load_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def category_from_label(llm_label: dict | None) -> str | None:
    if not isinstance(llm_label, dict):
        return None
    if llm_label.get('entity_ambiguous') == 'No':
        return 'No'
    if llm_label.get('entity_ambiguous') == 'Yes':
        return 'Yes'
    return None


def infer_entity_ambiguous_from_trajectory(row: dict) -> str | None:
    labels = []
    for step in row.get('trajectory') or []:
        if not isinstance(step, dict):
            continue
        category = category_from_label(step.get('llm_label'))
        if category in {'Yes', 'No'}:
            labels.append(category)
    if 'Yes' in labels:
        return 'Yes'
    if 'No' in labels:
        return 'No'
    return None


def build_binary_gptacc_rows():
    category_map = {}
    for label_path in sorted(ROOT.glob(LABEL_GLOB)):
        if 'first_round_oracle_rewrite' in str(label_path):
            continue
        for item in load_jsonl(label_path):
            annotation_id = item.get('annotation_id') or item.get('id')
            query_steps = item.get('query_steps')
            if query_steps is None:
                query_steps = [
                    step for step in item.get('trajectory', [])
                    if step.get('action') == 'search'
                ]
            sample_label = 'No'
            for step in query_steps:
                category = category_from_label(step.get('llm_label'))
                if category == 'Yes':
                    sample_label = 'Yes'
                    break
            category_map[annotation_id] = sample_label

    grouped = defaultdict(list)
    for intermediate_path in sorted(ROOT.glob(INTERMEDIATE_GLOB)):
        if 'first_round_oracle_rewrite' in str(intermediate_path):
            continue
        with intermediate_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        for row in data:
            sample_id = row['id']
            metric_score = ((row.get('output') or {}).get('metric_score') or {})
            if 'gpt_acc' not in metric_score or sample_id not in category_map:
                continue
            label = 'Entity\nambiguous' if category_map[sample_id] == 'Yes' else 'Non-entity\nambiguous'
            grouped[label].append(float(metric_score['gpt_acc']))

    rows = []
    for label in ['Entity\nambiguous', 'Non-entity\nambiguous']:
        values = grouped.get(label, [])
        rows.append(
            {
                'label': label,
                'mean': float(np.mean(np.array(values, dtype=float))) if values else 0.0,
                'count': len(values),
            }
        )
    return rows


def build_tasktype_rows_by_group():
    sample_info = {}
    for path in sorted(ROOT.glob(REWRITE_LABEL_GLOB)):
        if 'first_round_oracle_rewrite' in str(path):
            continue
        for row in load_jsonl(path):
            if (
                row.get('external_evidence_dependency') == 'Yes'
                and row.get('text_alone_identifies_entity') == 'No'
            ):
                sample_info[row['id']] = {
                    'task_type': row.get('task_type', 'Unknown'),
                    'entity_ambiguous': None,
                }

    for path in sorted(ROOT.glob(LABEL_GLOB)):
        if 'first_round_oracle_rewrite' in str(path):
            continue
        for row in load_jsonl(path):
            sample_id = row['id']
            if sample_id not in sample_info:
                continue
            sample_info[sample_id]['entity_ambiguous'] = infer_entity_ambiguous_from_trajectory(row)

    metric_map = {}
    for path in sorted(ROOT.glob(INTERMEDIATE_GLOB)):
        if 'first_round_oracle_rewrite' in str(path):
            continue
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        for row in data:
            metric_score = ((row.get('output') or {}).get('metric_score') or {})
            if 'gpt_acc' in metric_score:
                metric_map[row['id']] = float(metric_score['gpt_acc'])

    grouped = defaultdict(lambda: defaultdict(list))
    for sample_id, info in sample_info.items():
        entity_ambiguous = info['entity_ambiguous']
        task_type = info['task_type']
        if sample_id in metric_map and entity_ambiguous in ENTITY_GROUPS:
            grouped[entity_ambiguous][task_type].append(metric_map[sample_id])

    rows_by_group = {}
    for entity_ambiguous in ENTITY_GROUPS:
        rows = []
        for task_type in TASK_ORDER:
            values = grouped[entity_ambiguous].get(task_type, [])
            if not values:
                continue
            rows.append(
                {
                    'task_type': task_type,
                    'label': TASK_LABELS.get(task_type, task_type),
                    'mean': float(np.mean(np.array(values, dtype=float))),
                    'count': len(values),
                }
            )
        rows_by_group[entity_ambiguous] = rows
    return rows_by_group


def get_task_plot_order(rows_by_group):
    no_rows = {row['task_type']: row['mean'] for row in rows_by_group.get('No', [])}
    return sorted(
        TASK_ORDER,
        key=lambda task_type: (task_type not in no_rows, no_rows.get(task_type, float('inf'))),
    )


def plot_binary(ax) -> None:
    rows = build_binary_gptacc_rows()
    xs = np.arange(len(rows))
    means = np.array([row['mean'] for row in rows])
    color = '#A8D0E6'

    bars = ax.bar(
        xs,
        means,
        color=color,
        alpha=0.9,
        width=0.62,
    )
    ax.set_xticks(xs, [row['label'] for row in rows])
    ax.set_ylabel('LLM-as-Judge')
    ax.set_title('(a) Binary Entity Ambiguity')
    ax.grid(axis='y', linestyle='--', alpha=0.22)
    ax.set_ylim(0.0, max(means) + 0.08 if len(means) else 1.0)

    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row['mean'] + 0.012,
            f"{row['mean']:.3f}",
            ha='center',
            va='bottom',
            fontsize=13,
            color='#6B7A8F',
            bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.92),
        )


def plot_tasktype(ax) -> None:
    rows_by_group = build_tasktype_rows_by_group()
    plot_order = get_task_plot_order(rows_by_group)
    xs = np.arange(len(plot_order))
    width = 0.34
    ax.set_xticks(xs, [TASK_LABELS.get(task_type, task_type) for task_type in plot_order])
    ax.set_ylabel('LLM-as-Judge')
    ax.set_title('(b) Task-Type Stratification')
    ax.grid(axis='y', linestyle='--', alpha=0.22)
    max_y = 0.0

    for offset_idx, entity_ambiguous in enumerate(ENTITY_GROUPS):
        rows = rows_by_group.get(entity_ambiguous, [])
        style = ENTITY_STYLES[entity_ambiguous]
        row_map = {row['task_type']: row for row in rows}
        x_vals = []
        means = []
        for idx, task_type in enumerate(plot_order):
            row = row_map.get(task_type)
            if row is None:
                continue
            x_vals.append(idx)
            means.append(row['mean'])
            max_y = max(max_y, row['mean'])

        if not x_vals:
            continue

        x_arr = np.array(x_vals)
        mean_arr = np.array(means)
        bar_positions = x_arr + (-width / 2 if offset_idx == 0 else width / 2)
        bars = ax.bar(
            bar_positions,
            mean_arr,
            color=style['color'],
            alpha=0.92,
            width=width,
            label=style['label'],
        )
        for bar, mean, entity_group in zip(bars, mean_arr, [entity_ambiguous] * len(mean_arr)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.02,
                f'{mean:.3f}',
                ha='center',
                va='bottom',
                fontsize=13,
                color='#6B7A8F' if entity_group == 'Yes' else '#C46E2E',
            )

    ax.set_ylim(0.0, max(1.0, max_y + 0.08))
    ax.legend(frameon=False, loc='upper left')


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.5, 5.4),
        gridspec_kw={'width_ratios': [0.8, 1.45]},
        constrained_layout=True,
    )
    plot_binary(axes[0])
    plot_tasktype(axes[1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f'{OUT_NAME}.pdf'
    png_path = OUT_DIR / f'{OUT_NAME}.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == '__main__':
    main()
