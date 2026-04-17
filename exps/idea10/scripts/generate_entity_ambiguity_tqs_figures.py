#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path('/home/you/FlashRAG/exps/idea10/idea_reports/new')
BASELINE_METRIC_PATH = Path(
    '/home/you/FlashRAG/exps/idea10/data/result/'
    'crag_mm_2026_03_31_14_06_experiment/trajectory_quality_eval/trajectory_quality_samples.jsonl'
)
AMBIGUITY_LABEL_PATH = Path(
    '/home/you/FlashRAG/exps/idea10/data/result/'
    'crag_mm_2026_03_31_14_06_experiment/label/qwen/'
    'trajectory_annotation.llm_labeled.adjudicated.jsonl'
)
ORACLE_METRIC_PATH = Path(
    '/home/you/FlashRAG/exps/idea10/data/result/'
    '2026_04_10_12_53_48_first_round_oracle_rewrite_experiment/'
    'trajectory_quality_eval/trajectory_quality_samples.jsonl'
)

CATEGORY_ORDER = [
    'Object Identification',
    'Description',
    'Indirect Entity Ambiguity',
    'No Object Involved',
    'Multiple',
]
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 7
STATUS_ORDER = [
    ('Overall', None),
    ('Missing final\nanswer', 'missing_final_answer'),
    ('OK', 'ok'),
]
FINE_LABELS = {
    'No Object Involved': 'No Object\nInvolved',
    'Multiple': 'Multiple',
    'Object Identification': 'Object\nIdentification',
    'Description': 'Description',
    'Indirect Entity Ambiguity': 'Indirect\nEntity Ambiguity',
}


def style() -> None:
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 15,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'axes.spines.top': True,
            'axes.spines.right': True,
        }
    )


def load_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sample_level_category(record: dict) -> str:
    categories = []
    for step in record.get('query_steps', []):
        llm_label = step.get('llm_label') or {}
        if llm_label.get('entity_ambiguous') == 'Yes':
            level = llm_label.get('ambiguity_level') or 'No'
            if level != 'No':
                categories.append(level)
    unique_categories = list(dict.fromkeys(categories))
    if not unique_categories:
        return 'No'
    if len(unique_categories) == 1:
        return unique_categories[0]
    return 'Multiple'


def load_rte_scores(path: Path) -> dict[str, float]:
    return {row['id']: row['trajectory_quality_score'] for row in load_jsonl(path)}


def load_sample_categories(path: Path) -> dict[str, str]:
    return {row['annotation_id']: sample_level_category(row) for row in load_jsonl(path)}


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


def build_binary_stats():
    scores = load_rte_scores(BASELINE_METRIC_PATH)
    sample_categories = load_sample_categories(AMBIGUITY_LABEL_PATH)
    grouped = {
        'Entity\nambiguous': [],
        'Non-entity\nambiguous': [],
    }
    for sample_id, score in scores.items():
        category = sample_categories.get(sample_id, 'Unknown')
        if category == 'Unknown':
            continue
        bucket = 'Non-entity\nambiguous' if category == 'No' else 'Entity\nambiguous'
        grouped[bucket].append(float(score))

    rows = []
    for label in ['Entity\nambiguous', 'Non-entity\nambiguous']:
        mean, ci_low, ci_high = bootstrap_mean_ci(grouped[label])
        rows.append(
            {
                'label': label,
                'mean': mean,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'count': len(grouped[label]),
            }
        )
    return rows


def build_fine_stats():
    scores = load_rte_scores(BASELINE_METRIC_PATH)
    sample_categories = load_sample_categories(AMBIGUITY_LABEL_PATH)
    grouped = defaultdict(list)
    for sample_id, score in scores.items():
        category = sample_categories.get(sample_id, 'Unknown')
        if category in {'Unknown', 'No'}:
            continue
        grouped[category].append(float(score))

    rows = []
    for category in ['No Object Involved', 'Multiple', 'Object Identification', 'Description', 'Indirect Entity Ambiguity']:
        values = grouped.get(category, [])
        if not values:
            continue
        mean, ci_low, ci_high = bootstrap_mean_ci(values)
        rows.append(
            {
                'label': FINE_LABELS[category],
                'mean': mean,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'count': len(values),
            }
        )
    return rows


def build_ambiguous_category_counts():
    sample_categories = load_sample_categories(AMBIGUITY_LABEL_PATH)
    counts = defaultdict(int)
    for category in sample_categories.values():
        if category == 'No':
            continue
        counts[category] += 1
    return [(category, counts[category]) for category in CATEGORY_ORDER if counts.get(category)]


def build_oracle_rewrite_overall_stats():
    baseline_rows = {row['id']: row for row in load_jsonl(BASELINE_METRIC_PATH)}
    oracle_rows = {row['id']: row for row in load_jsonl(ORACLE_METRIC_PATH)}
    paired_ids = sorted(set(baseline_rows) & set(oracle_rows))

    rows = []
    for label, status in STATUS_ORDER:
        pairs = []
        for sample_id in paired_ids:
            baseline_row = baseline_rows[sample_id]
            oracle_row = oracle_rows[sample_id]
            if status is not None and baseline_row.get('status') != status:
                continue
            pairs.append(
                (
                    float(baseline_row['trajectory_quality_score']),
                    float(oracle_row['trajectory_quality_score']),
                )
            )
        rows.append(
            {
                'label': label,
                'count': len(pairs),
                'baseline': float(np.mean([item[0] for item in pairs])) if pairs else 0.0,
                'oracle': float(np.mean([item[1] for item in pairs])) if pairs else 0.0,
            }
        )
    return rows


def build_oracle_rewrite_category_stats():
    baseline_scores = load_rte_scores(BASELINE_METRIC_PATH)
    oracle_scores = load_rte_scores(ORACLE_METRIC_PATH)
    sample_categories = load_sample_categories(AMBIGUITY_LABEL_PATH)

    paired_ids = sorted(set(baseline_scores) & set(oracle_scores))
    grouped_pairs = defaultdict(list)
    for sample_id in paired_ids:
        category = sample_categories.get(sample_id, 'Unknown')
        if category == 'No':
            continue
        grouped_pairs[category].append((baseline_scores[sample_id], oracle_scores[sample_id]))

    rows = []
    for category in CATEGORY_ORDER:
        pairs = grouped_pairs.get(category)
        if not pairs:
            continue
        baseline_mean = float(np.mean([item[0] for item in pairs]))
        oracle_mean = float(np.mean([item[1] for item in pairs]))
        rows.append({
            'label': category,
            'count': len(pairs),
            'baseline': baseline_mean,
            'oracle': oracle_mean,
            'delta': oracle_mean - baseline_mean,
        })
    return rows


def plot_point_range(ax, data, title, color, rotation=0, label_offsets=None):
    xs = np.arange(len(data))
    means = np.array([row['mean'] for row in data])
    lowers = np.array([row['mean'] - row['ci_low'] for row in data])
    uppers = np.array([row['ci_high'] - row['mean'] for row in data])

    ax.errorbar(
        xs,
        means,
        yerr=[lowers, uppers],
        fmt='o',
        color=color,
        ecolor=color,
        elinewidth=2.2,
        capsize=5,
        markersize=8,
        zorder=3,
    )
    ax.plot(xs, means, color=color, linewidth=1.6, alpha=0.55, zorder=2)
    ax.set_xticks(xs, [row['label'] for row in data])
    ax.tick_params(axis='x', rotation=rotation)
    ax.set_ylabel('RTE')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.22)
    ax.set_ylim(0.0, max(row['ci_high'] for row in data) + 0.05)

    if label_offsets is None:
        label_offsets = [(0.08, 0.004, 'left', 'bottom') for _ in data]

    for x, row, (dx, dy, ha, va) in zip(xs, data, label_offsets):
        label_x = x + dx
        label_y = row['mean'] + dy
        ax.text(
            label_x,
            label_y,
            f"{row['mean']:.3f}",
            ha=ha,
            va=va,
            fontsize=13,
            color=color,
            bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.92),
        )


def make_ambiguity_figure():
    binary_stats = build_binary_stats()
    fine_stats = build_fine_stats()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.0), constrained_layout=True)
    plot_point_range(
        axes[0],
        binary_stats,
        '(a) Binary Entity Ambiguity',
        color='#4c78a8',
        rotation=0,
        label_offsets=[
            (0.04, -0.006, 'left', 'top'),
            (-0.04, -0.006, 'right', 'top'),
        ],
    )
    plot_point_range(
        axes[1],
        fine_stats,
        '(b) Fine-Grained Ambiguity Levels',
        color='#d95f02',
        rotation=16,
        label_offsets=[
            (0.257, 0.006, 'right', 'bottom'),
            (0.08, -0.006, 'left', 'top'),
            (0.08, -0.006, 'left', 'top'),
            (0.08, -0.006, 'left', 'top'),
            (-0.08, 0.006, 'right', 'bottom'),
        ],
    )
    return fig


def make_ambiguous_distribution_figure():
    ambiguous_category_counts = build_ambiguous_category_counts()
    fig, ax = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    labels = [item[0] for item in ambiguous_category_counts]
    counts = np.array([item[1] for item in ambiguous_category_counts])
    pcts = 100.0 * counts / counts.sum()
    ys = np.arange(len(labels))[::-1]
    colors = ['#E88C9A', '#F3B37A', '#F6C98E', '#8DB7E0', '#B9D4EE']

    ax.barh(ys, pcts, color=colors, alpha=0.92, height=0.56)
    ax.set_yticks(ys, labels)
    for tick in ax.get_yticklabels():
        tick.set_rotation(25)
        tick.set_rotation_mode('anchor')
        tick.set_ha('right')
        tick.set_va('center')
    ax.grid(axis='x', linestyle='--', alpha=0.22)

    xmax = pcts.max() * 1.22
    ax.set_xlim(0, xmax)
    xticks = [0, 10, 20, 30, 40, 50]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{x}%' for x in xticks])

    for y, pct in zip(ys, pcts):
        ax.text(
            pct + xmax * 0.015,
            y,
            f'{pct:.1f}%',
            va='center',
            ha='left',
            fontsize=10,
            color='#333333',
        )
    return fig


def make_oracle_rewrite_figure():
    oracle_rewrite_original = build_oracle_rewrite_overall_stats()
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ys = np.arange(len(oracle_rewrite_original))[::-1]

    for y, row in zip(ys, oracle_rewrite_original):
        base = row['baseline']
        oracle = row['oracle']
        line_color = '#2a9d8f' if oracle >= base else '#c0392b'
        ax.plot([base, oracle], [y, y], color=line_color, linewidth=2.5, alpha=0.85)
        ax.scatter(base, y, color='#4c78a8', s=65, zorder=3, label='Baseline' if y == ys[0] else None)
        ax.scatter(oracle, y, color='#f58518', s=65, zorder=3, label='Oracle Rewrite' if y == ys[0] else None)
        ax.text(base + 0.006, y + 0.03, f'{base:.3f}', ha='left', va='bottom', fontsize=9, color='#4c78a8')
        ax.text(oracle + 0.006, y + 0.03, f'{oracle:.3f}', ha='right', va='bottom', fontsize=9, color='#f58518')

    ax.set_yticks(ys, [row['label'] for row in oracle_rewrite_original])
    ax.set_xlabel('Mean RTE')
    ax.grid(axis='x', linestyle='--', alpha=0.25)
    ax.legend(frameon=False, loc='upper right')
    ax.set_xlim(0.0, 0.42)
    return fig


def make_oracle_rewrite_category_figure():
    oracle_rewrite = build_oracle_rewrite_category_stats()
    fig, ax = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)
    ys = np.arange(len(oracle_rewrite))[::-1]

    for y, row in zip(ys, oracle_rewrite):
        base = row['baseline']
        oracle = row['oracle']
        line_color = '#2a9d8f' if oracle >= base else '#c0392b'
        ax.plot([base, oracle], [y, y], color=line_color, linewidth=2.5, alpha=0.85)
        ax.scatter(base, y, color='#4c78a8', s=65, zorder=3, label='Baseline' if y == ys[0] else None)
        ax.scatter(oracle, y, color='#f58518', s=65, zorder=3, label='Oracle Rewrite' if y == ys[0] else None)
        ax.text(base + 0.005, y + 0.08, f'{base:.3f}', ha='left', va='bottom', fontsize=9, color='#4c78a8')
        ax.text(oracle + 0.005, y - 0.08, f'{oracle:.3f}', ha='left', va='top', fontsize=9, color='#f58518')

    ax.set_yticks(ys, [f"{row['label']}" for row in oracle_rewrite])
    for tick in ax.get_yticklabels():
        tick.set_rotation(25)
        tick.set_rotation_mode('anchor')
        tick.set_ha('right')
        tick.set_va('center')
    ax.set_xlabel('Mean RTE')
    ax.set_ylabel('')
    ax.grid(axis='x', linestyle='--', alpha=0.25)
    ax.legend(frameon=False, loc='lower right')
    max_value = max(max(row['baseline'], row['oracle']) for row in oracle_rewrite)
    ax.set_xlim(0.0, max(0.18, max_value + 0.06))
    return fig

def save(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f'{name}.pdf'
    png_path = OUTPUT_DIR / f'{name}.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    style()
    ambiguity_pdf, ambiguity_png = save(make_ambiguity_figure(), 'entity_ambiguity_rte_point_range')
    dist_pdf, dist_png = save(make_ambiguous_distribution_figure(), 'entity_ambiguity_category_distribution')
    oracle_pdf, oracle_png = save(make_oracle_rewrite_figure(), 'oracle_rewrite_rte_slope')
    oracle_cat_pdf, oracle_cat_png = save(make_oracle_rewrite_category_figure(), 'oracle_rewrite_rte_slope_by_category')
    print(ambiguity_pdf)
    print(ambiguity_png)
    print(dist_pdf)
    print(dist_png)
    print(oracle_pdf)
    print(oracle_png)
    print(oracle_cat_pdf)
    print(oracle_cat_png)


if __name__ == '__main__':
    main()
