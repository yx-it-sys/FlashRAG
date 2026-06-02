#!/usr/bin/env python3

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover
    pearsonr = None
    spearmanr = None


ROOT = Path('/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B')
OUTPUT_DIR = Path('/home/you/FlashRAG/exps/idea10/idea_reports/refamb_analysis_figs')
RTE_GLOB = 'RefAmb_*_refamb_*_stage/trajectory_quality_eval_whole_delta_F/trajectory_quality_samples.jsonl'
METRIC_GLOB = 'RefAmb_*_refamb_*_stage/intermediate_data.json'
RTE_CSV_PATH = OUTPUT_DIR / 'refamb_rte_metric_samples.csv'
CORR_CSV_PATH = OUTPUT_DIR / 'refamb_rte_metric_correlations.csv'
STRAT_CSV_PATH = OUTPUT_DIR / 'refamb_rte_metric_correlations_stratified.csv'
REPORT_MD_PATH = OUTPUT_DIR / 'refamb_rte_metric_correlation_report.md'
SCATTER_PDF_PATH = OUTPUT_DIR / 'refamb_rte_vs_metrics_scatter.pdf'
SCATTER_PNG_PATH = OUTPUT_DIR / 'refamb_rte_vs_metrics_scatter.png'

METRICS = ['em', 'acc', 'f1', 'recall', 'precision', 'gpt_acc']
DISPLAY_NAMES = {
    'em': 'EM',
    'acc': 'Acc',
    'f1': 'F1',
    'recall': 'Recall',
    'precision': 'Precision',
    'gpt_acc': 'GPTAcc',
}


def style() -> None:
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.04,
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


def load_rte_rows() -> dict[str, dict]:
    rows = {}
    for path in sorted(ROOT.glob(RTE_GLOB)):
        if 'first_round_oracle_rewrite' in str(path):
            continue
        for row in load_jsonl(path):
            rows[row['id']] = {
                'rte': float(row['trajectory_quality_score']),
                'status': row.get('status', ''),
                'num_iterations': int(row.get('num_iterations', 0)),
            }
    return rows


def load_metric_rows() -> dict[str, dict]:
    rows = {}
    for path in sorted(ROOT.glob(METRIC_GLOB)):
        if 'first_round_oracle_rewrite' in str(path):
            continue
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        for row in data:
            metric_score = ((row.get('output') or {}).get('metric_score') or {})
            if not metric_score:
                continue
            rows[row['id']] = {
                'source': row.get('source', ''),
                'task_type': row.get('task_type', ''),
                'metrics': {metric: float(metric_score.get(metric, 0.0)) for metric in METRICS},
            }
    return rows


def build_samples() -> list[dict]:
    rte_rows = load_rte_rows()
    metric_rows = load_metric_rows()
    sample_ids = sorted(set(rte_rows) & set(metric_rows))
    samples = []
    for sample_id in sample_ids:
        rte_row = rte_rows[sample_id]
        metric_row = metric_rows[sample_id]
        row = {
            'id': sample_id,
            'source': metric_row['source'],
            'task_type': metric_row['task_type'],
            'status': rte_row['status'],
            'num_iterations': rte_row['num_iterations'],
            'rte': rte_row['rte'],
        }
        row.update(metric_row['metrics'])
        samples.append(row)
    return samples


def compute_corr(x_vals: list[float], y_vals: list[float]) -> dict[str, float]:
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    result = {
        'n': int(len(x)),
        'pearson_r': float('nan'),
        'pearson_p': float('nan'),
        'spearman_rho': float('nan'),
        'spearman_p': float('nan'),
    }
    if len(x) < 2:
        return result

    if pearsonr is not None:
        pr = pearsonr(x, y)
        result['pearson_r'] = float(pr.statistic)
        result['pearson_p'] = float(pr.pvalue)
    else:  # pragma: no cover
        result['pearson_r'] = float(np.corrcoef(x, y)[0, 1])

    if spearmanr is not None:
        sr = spearmanr(x, y)
        result['spearman_rho'] = float(sr.statistic)
        result['spearman_p'] = float(sr.pvalue)
    else:  # pragma: no cover
        x_rank = np.argsort(np.argsort(x))
        y_rank = np.argsort(np.argsort(y))
        result['spearman_rho'] = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return result


def compute_overall_correlations(samples: list[dict]) -> list[dict]:
    rows = []
    x_vals = [sample['rte'] for sample in samples]
    for metric in METRICS:
        stats = compute_corr(x_vals, [sample[metric] for sample in samples])
        rows.append({'metric': metric, **stats})
    return rows


def compute_stratified_correlations(samples: list[dict], field: str) -> list[dict]:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample[field]].append(sample)

    rows = []
    for group_name, group_samples in sorted(grouped.items()):
        x_vals = [sample['rte'] for sample in group_samples]
        for metric in METRICS:
            stats = compute_corr(x_vals, [sample[metric] for sample in group_samples])
            rows.append(
                {
                    'group_field': field,
                    'group_name': group_name,
                    'metric': metric,
                    **stats,
                }
            )
    return rows


def write_sample_csv(samples: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ['id', 'source', 'task_type', 'status', 'num_iterations', 'rte', *METRICS]
    with RTE_CSV_PATH.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def write_corr_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'nan'
    return f'{value:.4f}'


def write_report(samples: list[dict], overall: list[dict], stratified: list[dict]) -> None:
    by_source = defaultdict(int)
    by_task = defaultdict(int)
    by_status = defaultdict(int)
    for sample in samples:
        by_source[sample['source']] += 1
        by_task[sample['task_type']] += 1
        by_status[sample['status']] += 1

    top_overall = sorted(
        overall,
        key=lambda row: abs(row['spearman_rho']) if not math.isnan(row['spearman_rho']) else -1.0,
        reverse=True,
    )

    lines = [
        '# RTE vs Generation-Metric Correlation',
        '',
        f'- Samples: {len(samples)}',
        f'- Sources: {dict(sorted(by_source.items()))}',
        f'- Task types: {dict(sorted(by_task.items()))}',
        f'- Status counts: {dict(sorted(by_status.items()))}',
        '',
        '## Overall Correlations',
        '',
        '| Metric | n | Pearson r | Pearson p | Spearman rho | Spearman p |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in overall:
        lines.append(
            f"| {DISPLAY_NAMES[row['metric']]} | {row['n']} | {safe_fmt(row['pearson_r'])} | {safe_fmt(row['pearson_p'])} | {safe_fmt(row['spearman_rho'])} | {safe_fmt(row['spearman_p'])} |"
        )

    lines.extend(
        [
            '',
            '## Key Findings',
            '',
        ]
    )
    for row in top_overall[:3]:
        lines.append(
            f"1. `RTE` vs `{DISPLAY_NAMES[row['metric']]}`: Spearman rho = {safe_fmt(row['spearman_rho'])}, Pearson r = {safe_fmt(row['pearson_r'])}, n = {row['n']}."
        )

    lines.extend(
        [
            '',
            '## Stratified Correlations',
            '',
            '| Group Field | Group Name | Metric | n | Spearman rho | Spearman p |',
            '| --- | --- | --- | ---: | ---: | ---: |',
        ]
    )
    for row in stratified:
        lines.append(
            f"| {row['group_field']} | {row['group_name']} | {DISPLAY_NAMES[row['metric']]} | {row['n']} | {safe_fmt(row['spearman_rho'])} | {safe_fmt(row['spearman_p'])} |"
        )

    lines.extend(
        [
            '',
            '## Suggested Next Checks',
            '',
            '1. Compare these correlations within `OK`, `missing_final_answer`, and `generation_error` separately when interpreting process-quality vs answer-quality coupling.',
            '2. If needed, add regression controls for `source` and `task_type` to test whether the `RTE` association remains after stratification.',
        ]
    )
    REPORT_MD_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def add_fit_line(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 2:
        return
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if abs(x_max - x_min) < 1e-12:
        return
    coeffs = np.polyfit(x, y, deg=1)
    x_line = np.linspace(x_min, x_max, 100)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax.plot(x_line, y_line, color=color, linewidth=1.8, alpha=0.9)


def draw_scatter(samples: list[dict], overall: list[dict]) -> None:
    style()
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)
    color = '#2F6C8F'
    x = np.array([sample['rte'] for sample in samples], dtype=float)
    overall_map = {row['metric']: row for row in overall}

    for ax, metric in zip(axes.flat, METRICS):
        y = np.array([sample[metric] for sample in samples], dtype=float)
        ax.scatter(x, y, s=18, alpha=0.55, color=color, edgecolors='none')
        add_fit_line(ax, x, y, color='#F4A261')
        rho = overall_map[metric]['spearman_rho']
        pval = overall_map[metric]['spearman_p']
        ax.set_title(f"{DISPLAY_NAMES[metric]}  (rho={safe_fmt(rho)}, p={safe_fmt(pval)})")
        ax.set_xlabel('RTE')
        ax.set_ylabel(DISPLAY_NAMES[metric])
        ax.grid(True, linestyle='--', alpha=0.22)

    fig.savefig(SCATTER_PDF_PATH)
    fig.savefig(SCATTER_PNG_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    samples = build_samples()
    if not samples:
        raise RuntimeError('No aligned RTE/metric samples found.')

    overall = compute_overall_correlations(samples)
    stratified = []
    for field in ['source', 'task_type', 'status']:
        stratified.extend(compute_stratified_correlations(samples, field))

    write_sample_csv(samples)
    write_corr_csv(
        overall,
        CORR_CSV_PATH,
        ['metric', 'n', 'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p'],
    )
    write_corr_csv(
        stratified,
        STRAT_CSV_PATH,
        ['group_field', 'group_name', 'metric', 'n', 'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p'],
    )
    write_report(samples, overall, stratified)
    draw_scatter(samples, overall)

    print(RTE_CSV_PATH)
    print(CORR_CSV_PATH)
    print(STRAT_CSV_PATH)
    print(REPORT_MD_PATH)
    print(SCATTER_PDF_PATH)
    print(SCATTER_PNG_PATH)


if __name__ == '__main__':
    main()
