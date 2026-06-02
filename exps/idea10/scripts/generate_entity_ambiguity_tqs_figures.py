#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path('/home/you/FlashRAG/exps/idea10/idea_reports/refamb_analysis_figs')
REFAMB_RESULT_DIR = Path(
    '/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B'
)
REFAMB_BASELINE_METRIC_GLOB = (
    'RefAmb_*_refamb_*_stage/trajectory_quality_eval_whole_delta_F/'
    'trajectory_quality_samples.jsonl'
)
REFAMB_ORACLE_METRIC_GLOB = (
    'RefAmb_*_refamb_*_stage/first_round_oracle_rewrite/'
    'trajectory_quality_eval_whole_delta_F/trajectory_quality_samples.jsonl'
)
REFAMB_AMBIGUITY_LABEL_GLOB = (
    'RefAmb_*_refamb_*_stage/label/deepseek/'
    'omnisearch_trajectories.entity_ambiguity_labeled.jsonl'
)
BASELINE_METRIC_PATH = Path(
    '/home/you/FlashRAG/exps/idea10/data/result/'
    'crag_mm_2026_03_31_14_06_experiment/trajectory_quality_final_use_whole_delta_f/trajectory_quality_samples.jsonl'
)
AMBIGUITY_LABEL_PATH = Path(
    '/home/you/FlashRAG/exps/idea10/data/result/'
    'crag_mm_2026_03_31_14_06_experiment/label/qwen/'
    'trajectory_annotation.llm_labeled.adjudicated.jsonl'
)
ORACLE_METRIC_PATH = Path(
    '/home/you/FlashRAG/exps/idea10/data/result/'
    '2026_04_10_12_53_48_first_round_oracle_rewrite_experiment/'
    'trajectory_quality_eval_whole_deltaf/trajectory_quality_samples.jsonl'
)

CATEGORY_ORDER = [
    'Object Identification',
    'Description',
    'Indirect Entity Ambiguity',
    'No Object Involved',
    'Mixed',
]
AMBIGUITY_PRIORITY = [
    'Object Identification',
    'Description',
    'Indirect Entity Ambiguity',
    'No Object Involved',
]
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 7
STATUS_ORDER = [
    ('Overall', None),
    ('Missing final\nanswer', 'missing_final_answer'),
    ('Generation\nError', 'generation_error'),
    ('OK', 'ok'),
]
FINE_LABELS = {
    'No Object Involved': 'No Obj.',
    'Mixed': 'Mixed',
    'Multiple': 'Multiple',
    'Object Identification': 'OI.',
    'Description': 'Desc.',
    'Indirect Entity Ambiguity': 'IEA.',
    'Object Identification + Description': 'OI.+Desc.',
    'Object Identification + Indirect Entity Ambiguity': 'OI.+IEA.',
    'Description + Indirect Entity Ambiguity': 'Desc.+IEA.',
}

MIXED_COMBO_LABELS = {
    ('Object Identification', 'Description'): 'Object Identification + Description',
    ('Object Identification', 'Indirect Entity Ambiguity'): 'Object Identification + Indirect Entity Ambiguity',
    ('Description', 'Indirect Entity Ambiguity'): 'Description + Indirect Entity Ambiguity',
}


def canonical_combo_label(categories: list[str]) -> str:
    unique = []
    seen = set()
    for category in categories:
        if category not in seen:
            unique.append(category)
            seen.add(category)
    if len(unique) <= 1:
        return unique[0] if unique else 'No'
    ordered = [cat for cat in AMBIGUITY_PRIORITY if cat in seen]
    return MIXED_COMBO_LABELS.get(tuple(ordered), ' + '.join(ordered))


def style() -> None:
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 17,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 15,
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
    query_steps = record.get('query_steps')
    if query_steps is None:
        query_steps = [
            step for step in record.get('trajectory', [])
            if step.get('action') == 'search'
        ]
    for step in query_steps:
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
    return canonical_combo_label(unique_categories)


def load_rte_scores(path: Path) -> dict[str, float]:
    return {row['id']: row['trajectory_quality_score'] for row in load_jsonl(path)}


def load_sample_categories(path: Path) -> dict[str, str]:
    return {row['annotation_id']: sample_level_category(row) for row in load_jsonl(path)}


def load_refamb_rte_scores() -> dict[str, float]:
    scores = {}
    for path in sorted(REFAMB_RESULT_DIR.glob(REFAMB_BASELINE_METRIC_GLOB)):
        scores.update(load_rte_scores(path))
    return scores


def load_refamb_oracle_rte_scores() -> dict[str, float]:
    scores = {}
    for path in sorted(REFAMB_RESULT_DIR.glob(REFAMB_ORACLE_METRIC_GLOB)):
        scores.update(load_rte_scores(path))
    return scores


def load_refamb_sample_categories() -> dict[str, str]:
    categories = {}
    for path in sorted(REFAMB_RESULT_DIR.glob(REFAMB_AMBIGUITY_LABEL_GLOB)):
        categories.update({row['id']: sample_level_category(row) for row in load_jsonl(path)})
    return categories


def load_refamb_baseline_metric_rows() -> dict[str, dict]:
    rows = {}
    for path in sorted(REFAMB_RESULT_DIR.glob(REFAMB_BASELINE_METRIC_GLOB)):
        for row in load_jsonl(path):
            rows[row['id']] = row
    return rows


def load_refamb_oracle_metric_rows() -> dict[str, dict]:
    rows = {}
    for path in sorted(REFAMB_RESULT_DIR.glob(REFAMB_ORACLE_METRIC_GLOB)):
        for row in load_jsonl(path):
            rows[row['id']] = row
    return rows


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
    scores = load_refamb_rte_scores()
    sample_categories = load_refamb_sample_categories()
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
    scores = load_refamb_rte_scores()
    sample_categories = load_refamb_sample_categories()
    grouped = defaultdict(list)
    for sample_id, score in scores.items():
        category = sample_categories.get(sample_id, 'Unknown')
        if category in {'Unknown', 'No'}:
            continue
        grouped[category].append(float(score))

    rows = []
    fine_categories = [
        'No Object Involved',
        'Object Identification',
        'Description',
        'Indirect Entity Ambiguity',
        'Object Identification + Description',
        'Object Identification + Indirect Entity Ambiguity',
        'Description + Indirect Entity Ambiguity',
    ]
    for category in fine_categories:
        values = grouped.get(category, [])
        if not values:
            continue
        mean, ci_low, ci_high = bootstrap_mean_ci(values)
        rows.append(
            {
                'label': FINE_LABELS.get(category, category.replace(' + ', '\n+\n')),
                'mean': mean,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'count': len(values),
            }
        )
    rows.sort(key=lambda row: row['mean'], reverse=True)
    return rows


def build_ambiguous_category_counts():
    sample_categories = load_refamb_sample_categories()
    counts = defaultdict(int)
    for category in sample_categories.values():
        if category == 'No':
            continue
        counts[category] += 1
    ordered = [
        'Object Identification',
        'Description',
        'Indirect Entity Ambiguity',
        'No Object Involved',
        'Object Identification + Description',
        'Object Identification + Indirect Entity Ambiguity',
        'Description + Indirect Entity Ambiguity',
        'Multiple',
    ]
    return [(category, counts[category]) for category in ordered if counts.get(category)]


def build_oracle_rewrite_overall_stats():
    baseline_rows = load_refamb_baseline_metric_rows()
    oracle_rows = load_refamb_oracle_metric_rows()
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
    baseline_scores = load_refamb_rte_scores()
    oracle_scores = load_refamb_oracle_rte_scores()
    sample_categories = load_refamb_sample_categories()

    paired_ids = sorted(set(baseline_scores) & set(oracle_scores))
    grouped_pairs = defaultdict(list)
    for sample_id in paired_ids:
        category = sample_categories.get(sample_id, 'Unknown')
        if category == 'No':
            continue
        if category in {
            'Object Identification + Description',
            'Object Identification + Indirect Entity Ambiguity',
            'Description + Indirect Entity Ambiguity',
        }:
            category = 'Mixed'
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


def plot_point_range(
    ax,
    data,
    title,
    color,
    rotation=0,
    label_offsets=None,
    xtick_fontsize=None,
    manual_xtick_labels=False,
):
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
    ax.plot(xs, means, color=color, linewidth=1.9, alpha=0.6, zorder=2)
    ax.set_xticks(xs)
    if manual_xtick_labels:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=4)
        for x, row in zip(xs, data):
            ax.text(
                x,
                0.01,
                row['label'],
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                fontsize=xtick_fontsize if xtick_fontsize is not None else 10,
                rotation=0,
                clip_on=False,
            )
    else:
        ax.set_xticklabels([row['label'] for row in data])
        ax.tick_params(axis='x', rotation=rotation)
        if xtick_fontsize is not None:
            ax.tick_params(axis='x', labelsize=xtick_fontsize)
        if rotation:
            for tick in ax.get_xticklabels():
                tick.set_rotation(rotation)
                tick.set_rotation_mode('anchor')
                tick.set_ha('right')
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


def plot_bar_chart(
    ax,
    data,
    title,
    color,
    rotation=0,
    xtick_fontsize=None,
    manual_xtick_labels=False,
):
    xs = np.arange(len(data))
    means = np.array([row['mean'] for row in data])
    lowers = np.array([row['mean'] - row['ci_low'] for row in data])
    uppers = np.array([row['ci_high'] - row['mean'] for row in data])

    bars = ax.bar(
        xs,
        means,
        color=color,
        alpha=0.88,
        width=0.62,
        yerr=[lowers, uppers],
        capsize=5,
        ecolor=color,
        error_kw={'elinewidth': 2.0},
    )
    ax.set_xticks(xs)
    if manual_xtick_labels:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=4)
        for x, row in zip(xs, data):
            ax.text(
                x,
                0.01,
                row['label'],
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                fontsize=xtick_fontsize if xtick_fontsize is not None else 10,
                rotation=0,
                clip_on=False,
            )
    else:
        ax.set_xticklabels([row['label'] for row in data])
        ax.tick_params(axis='x', rotation=rotation)
        if xtick_fontsize is not None:
            ax.tick_params(axis='x', labelsize=xtick_fontsize)
        if rotation:
            for tick in ax.get_xticklabels():
                tick.set_rotation(rotation)
                tick.set_rotation_mode('anchor')
                tick.set_ha('right')
    ax.set_ylabel('RTE')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.22)
    ax.set_ylim(0.0, max(row['ci_high'] for row in data) + 0.06)

    for bar, row in zip(bars, data):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row['mean'] + 0.008,
            f"{row['mean']:.3f}",
            ha='center',
            va='bottom',
            fontsize=13,
            color=color,
            bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.92),
        )


def make_ambiguity_figure():
    binary_stats = build_binary_stats()
    fine_stats = build_fine_stats()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(19.5, 6.2),
        gridspec_kw={'width_ratios': [0.85, 1.65]},
        constrained_layout=True,
    )
    plot_bar_chart(
        axes[0],
        binary_stats,
        '(a) Binary Entity Ambiguity',
        color='#A8D0E6',
        rotation=0,
    )
    plot_bar_chart(
        axes[1],
        fine_stats,
        '(b) Fine-Grained Ambiguity Levels',
        color='#F4A261',
        rotation=22,
        xtick_fontsize=15,
        manual_xtick_labels=False,
    )
    return fig


def make_binary_ambiguity_figure():
    binary_stats = build_binary_stats()
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    plot_bar_chart(
        ax,
        binary_stats,
        'Binary Entity Ambiguity',
        color='#A8D0E6',
        rotation=0,
    )
    return fig


def make_fine_ambiguity_figure():
    fine_stats = build_fine_stats()
    fig, ax = plt.subplots(figsize=(10.8, 5.4), constrained_layout=True)
    plot_bar_chart(
        ax,
        fine_stats,
        'Fine-Grained Ambiguity Levels',
        color='#F4A261',
        rotation=22,
        xtick_fontsize=15,
        manual_xtick_labels=False,
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
    xs = np.arange(len(oracle_rewrite_original))
    width = 0.34
    baseline_vals = np.array([row['baseline'] for row in oracle_rewrite_original])
    oracle_vals = np.array([row['oracle'] for row in oracle_rewrite_original])

    base_bars = ax.bar(
        xs - width / 2,
        baseline_vals,
        width=width,
        color='#CFE8F3',
        label='Baseline',
        alpha=0.95,
    )
    oracle_bars = ax.bar(
        xs + width / 2,
        oracle_vals,
        width=width,
        color='#F4A261',
        label='Oracle Rewrite',
        alpha=0.95,
    )

    ax.set_xticks(xs, [row['label'] for row in oracle_rewrite_original])
    ax.set_ylabel('Mean RTE')
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend(frameon=False, loc='upper left')
    max_value = max(
        max(row['baseline'], row['oracle'])
        for row in oracle_rewrite_original
    ) if oracle_rewrite_original else 0.0
    ax.set_ylim(0.0, max(0.42, max_value + 0.08))

    for bars, color in [(base_bars, '#6B7A8F'), (oracle_bars, '#C46E2E')]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.008,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                color=color,
            )
    return fig


def make_oracle_rewrite_category_figure():
    oracle_rewrite = build_oracle_rewrite_category_stats()
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    xs = np.arange(len(oracle_rewrite))
    width = 0.34
    baseline_vals = np.array([row['baseline'] for row in oracle_rewrite])
    oracle_vals = np.array([row['oracle'] for row in oracle_rewrite])

    base_bars = ax.bar(
        xs - width / 2,
        baseline_vals,
        width=width,
        color='#CFE8F3',
        label='Baseline',
        alpha=0.95,
    )
    oracle_bars = ax.bar(
        xs + width / 2,
        oracle_vals,
        width=width,
        color='#F4A261',
        label='Oracle Rewrite',
        alpha=0.95,
    )

    ax.set_xticks(xs, [row['label'] for row in oracle_rewrite])
    ax.tick_params(axis='x', rotation=22)
    for tick in ax.get_xticklabels():
        tick.set_rotation(22)
        tick.set_rotation_mode('anchor')
        tick.set_ha('right')
    ax.set_ylabel('Mean RTE')
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend(frameon=False, loc='upper left')
    max_value = max(max(row['baseline'], row['oracle']) for row in oracle_rewrite)
    ax.set_ylim(0.0, max(0.18, max_value + 0.08))

    for bars, color in [(base_bars, '#6B7A8F'), (oracle_bars, '#C46E2E')]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.008,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                color=color,
            )
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
