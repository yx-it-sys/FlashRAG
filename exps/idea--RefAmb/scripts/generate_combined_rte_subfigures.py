#!/usr/bin/env python3

from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path('/home/you/FlashRAG/exps/idea10/idea_reports/refamb_analysis_figs')
OUT_NAME = 'combined_binary_and_tasktype_rte'


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


def plot_binary(ax, entity_module) -> None:
    rows = entity_module.build_binary_stats()
    xs = np.arange(len(rows))
    means = np.array([row['mean'] for row in rows])
    lowers = np.array([row['mean'] - row['ci_low'] for row in rows])
    uppers = np.array([row['ci_high'] - row['mean'] for row in rows])
    color = '#A8D0E6'

    bars = ax.bar(
        xs,
        means,
        color=color,
        alpha=0.9,
        width=0.62,
        yerr=[lowers, uppers],
        capsize=5,
        ecolor=color,
        error_kw={'elinewidth': 2.0},
    )
    ax.set_xticks(xs, [row['label'] for row in rows])
    ax.set_ylabel('RTE')
    ax.set_title('(a) Binary Entity Ambiguity')
    ax.grid(axis='y', linestyle='--', alpha=0.22)
    ax.set_ylim(0.0, max(row['ci_high'] for row in rows) + 0.06)

    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row['mean'] + 0.008,
            f"{row['mean']:.3f}",
            ha='center',
            va='bottom',
            fontsize=13,
            color='#6B7A8F',
            bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.92),
        )


def plot_tasktype(ax, task_module) -> None:
    rows_by_group = task_module.build_rows()
    plot_order = task_module.get_plot_order(rows_by_group)
    xs = np.arange(len(plot_order))
    width = 0.34

    ax.set_xticks(xs, [task_module.TASK_LABELS.get(task_type, task_type) for task_type in plot_order])
    ax.set_ylabel('RTE')
    ax.set_title('(b) Task-Type Stratification')
    ax.grid(axis='y', linestyle='--', alpha=0.22)

    max_y = 0.0
    for offset_idx, entity_ambiguous in enumerate(task_module.ENTITY_GROUPS):
        rows = rows_by_group.get(entity_ambiguous, [])
        style = task_module.ENTITY_STYLES[entity_ambiguous]
        row_map = {row['task_type']: row for row in rows}

        x_vals = []
        means = []
        labels = []
        for idx, task_type in enumerate(plot_order):
            row = row_map.get(task_type)
            if row is None:
                continue
            x_vals.append(idx)
            means.append(row['display_mean'])
            labels.append(task_type)
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

        for bar, task_type, mean in zip(bars, labels, mean_arr):
            vertical_align = 'bottom'
            offset = 0.012
            bar_override = task_module.LABEL_POSITION_OVERRIDES.get((entity_ambiguous, task_type, 'bar'))
            point_override = task_module.LABEL_POSITION_OVERRIDES.get((entity_ambiguous, task_type))
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
                color='#6B7A8F' if entity_ambiguous == 'Yes' else '#C46E2E',
            )

    ax.set_ylim(0.0, max_y + 0.05)
    ax.legend(frameon=False, loc='upper left')


def main() -> None:
    apply_style()
    entity_module = load_module(
        '/home/you/FlashRAG/exps/idea10/scripts/generate_entity_ambiguity_tqs_figures.py',
        'entity_figs',
    )
    task_module = load_module(
        '/home/you/FlashRAG/exps/idea10/scripts/generate_external_text_yes_task_type_rte_figure.py',
        'task_figs',
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.5, 5.4),
        gridspec_kw={'width_ratios': [0.8, 1.45]},
        constrained_layout=True,
    )
    plot_binary(axes[0], entity_module)
    plot_tasktype(axes[1], task_module)

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
