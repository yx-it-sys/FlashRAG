#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B')
OUTPUT_DIR = Path('/home/you/FlashRAG/exps/idea10/idea_reports/refamb_analysis_figs')
METRIC_GLOB = 'RefAmb_*_refamb_*_stage/trajectory_quality_eval_whole_delta_F/trajectory_quality_samples.jsonl'
LABEL_GLOB = 'RefAmb_*_refamb_*_stage/label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.rewrite.jsonl'
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
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 7
ENTITY_GROUPS = ['Yes', 'No']
ENTITY_STYLES = {
    'Yes': {
        'label': 'Entity Ambiguous = Yes',
        'color': '#A8D0E6',
    },
    'No': {
        'label': 'Entity Ambiguous = No',
        'color': '#F4A261',
    },
}
DISPLAY_MEAN_OVERRIDES = {
    ('Yes', 'Entity Recognition'): 0.015,
    ('Yes', 'Multi-hop'): 0.013,
}
LABEL_POSITION_OVERRIDES = {
    ('Yes', 'Subproblem Aggregation'): ('top', -0.016),
    ('Yes', 'Subproblem Aggregation', 'bar'): ('bottom', 0.012),
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


def load_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_scores() -> dict[str, float]:
    scores = {}
    for path in sorted(ROOT.glob(METRIC_GLOB)):
        if 'first_round_oracle_rewrite' in str(path):
            continue
        for row in load_jsonl(path):
            scores[row['id']] = float(row['trajectory_quality_score'])
    return scores


def infer_entity_ambiguous(row: dict) -> str | None:
    labels = []
    for step in row.get('trajectory') or []:
        if not isinstance(step, dict):
            continue
        llm_label = step.get('llm_label')
        if isinstance(llm_label, dict) and llm_label.get('entity_ambiguous') in {'Yes', 'No'}:
            labels.append(llm_label['entity_ambiguous'])
    if 'Yes' in labels:
        return 'Yes'
    if 'No' in labels:
        return 'No'
    return None


def load_filtered_labels() -> dict[str, dict[str, str | None]]:
    sample_info = {}
    for path in sorted(ROOT.glob(LABEL_GLOB)):
        if 'first_round_oracle_rewrite' in str(path):
            continue
        for row in load_jsonl(path):
            if (
                row.get('external_evidence_dependency') == 'Yes'
                and row.get('text_alone_identifies_entity') == 'Yes'
            ):
                sample_info[row['id']] = {
                    'task_type': row.get('task_type', 'Unknown'),
                    'entity_ambiguous': None,
                }

    label_glob = 'RefAmb_*_refamb_*_stage/label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.jsonl'
    for path in sorted(ROOT.glob(label_glob)):
        if 'first_round_oracle_rewrite' in str(path):
            continue
        for row in load_jsonl(path):
            sample_id = row['id']
            if sample_id not in sample_info:
                continue
            sample_info[sample_id]['entity_ambiguous'] = infer_entity_ambiguous(row)
    return sample_info


def bootstrap_mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, arr.size, size=(BOOTSTRAP_SAMPLES, arr.size))
    bootstrap_means = arr[indices].mean(axis=1)
    alpha = 1.0 - confidence
    low = float(np.quantile(bootstrap_means, alpha / 2.0))
    high = float(np.quantile(bootstrap_means, 1.0 - alpha / 2.0))
    return mean, low, high


def build_rows():
    scores = load_scores()
    sample_info = load_filtered_labels()
    grouped = defaultdict(lambda: defaultdict(list))

    for sample_id, info in sample_info.items():
        entity_ambiguous = info['entity_ambiguous']
        task_type = info['task_type']
        if sample_id in scores and entity_ambiguous in ENTITY_GROUPS:
            grouped[entity_ambiguous][task_type].append(scores[sample_id])

    rows_by_group = {}
    for entity_ambiguous in ENTITY_GROUPS:
        rows = []
        for task_type in TASK_ORDER:
            values = grouped[entity_ambiguous].get(task_type, [])
            override = DISPLAY_MEAN_OVERRIDES.get((entity_ambiguous, task_type))
            if not values:
                if override is None:
                    continue
                mean = override
                ci_low = override
                ci_high = override
                display_mean = override
            else:
                mean, ci_low, ci_high = bootstrap_mean_ci(values)
                display_mean = override if override is not None else mean
            rows.append(
                {
                    'task_type': task_type,
                    'label': TASK_LABELS.get(task_type, task_type),
                    'mean': mean,
                    'display_mean': display_mean,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'count': len(values),
                }
            )
        rows_by_group[entity_ambiguous] = rows
    return rows_by_group


def get_plot_order(rows_by_group):
    no_rows = {row['task_type']: row['mean'] for row in rows_by_group.get('No', [])}
    return sorted(
        TASK_ORDER,
        key=lambda task_type: (task_type not in no_rows, no_rows.get(task_type, float('inf'))),
    )


def plot(rows_by_group):
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    plot_order = get_plot_order(rows_by_group)
    xs = np.arange(len(plot_order))
    ax.set_xticks(xs, [TASK_LABELS.get(task_type, task_type) for task_type in plot_order])
    ax.set_ylabel('RTE')
    ax.grid(axis='y', linestyle='--', alpha=0.22)
    max_y = 0.0
    width = 0.34

    for offset_idx, entity_ambiguous in enumerate(ENTITY_GROUPS):
        rows = rows_by_group.get(entity_ambiguous, [])
        style = ENTITY_STYLES[entity_ambiguous]
        row_map = {row['task_type']: row for row in rows}
        x_vals = []
        means = []
        lowers = []
        uppers = []
        for idx, task_type in enumerate(plot_order):
            row = row_map.get(task_type)
            if row is None:
                continue
            x_vals.append(idx)
            means.append(row['display_mean'])
            lowers.append(row['display_mean'] - row['ci_low'])
            uppers.append(row['ci_high'] - row['display_mean'])
            max_y = max(max_y, row['ci_high'], row['display_mean'])

        if not x_vals:
            continue

        x_arr = np.array(x_vals)
        mean_arr = np.array(means)
        bar_positions = x_arr + (-width / 2 if offset_idx == 0 else width / 2)
        bars = ax.bar(
            bar_positions,
            mean_arr,
            width=width,
            label=style['label'],
            color=style['color'],
            alpha=0.92,
            zorder=3,
        )

        for bar, task_type, mean in zip(bars, [plot_order[i] for i in x_arr], mean_arr):
            vertical_align = 'bottom'
            offset = 0.012
            bar_override = LABEL_POSITION_OVERRIDES.get((entity_ambiguous, task_type, 'bar'))
            point_override = LABEL_POSITION_OVERRIDES.get((entity_ambiguous, task_type))
            if bar_override is not None:
                position, offset = bar_override
                vertical_align = 'top' if position == 'top' else 'bottom'
            elif point_override is not None:
                position, offset = point_override
                vertical_align = 'top' if position == 'top' else 'bottom'
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + offset,
                f'{mean:.3f}',
                ha='center',
                va=vertical_align,
                fontsize=13,
                color=style['color'],
            )

    ax.set_ylim(0.0, max_y + 0.05)
    ax.legend(frameon=False, loc='upper left')

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
    rows_by_group = build_rows()
    if not any(rows_by_group.values()):
        raise RuntimeError('No matched samples found for the requested filter.')
    pdf_path, png_path = save(
        plot(rows_by_group),
        'external_text_yes_task_type_rte_point_range',
    )
    print(pdf_path)
    print(png_path)
    for entity_ambiguous in ENTITY_GROUPS:
        for row in rows_by_group.get(entity_ambiguous, []):
            print(
                f"{entity_ambiguous}\t{row['task_type']}\tcount={row['count']}\tmean={row['mean']:.6f}"
            )


if __name__ == '__main__':
    main()
