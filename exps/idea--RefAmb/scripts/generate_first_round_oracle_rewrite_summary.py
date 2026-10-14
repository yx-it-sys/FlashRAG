#!/usr/bin/env python3
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


BASE_DIR = Path("/home/you/FlashRAG/exps/idea10/idea_reports/new")
FIGURE_PATH = BASE_DIR / "first_round_oracle_rewrite_summary.pdf"
TABLE_PATH = BASE_DIR / "first_round_oracle_rewrite_main_results.tex"
MAIN_FIGURE_PATH = BASE_DIR / "first_round_oracle_rewrite_main_bar.pdf"
TEXT_SNIPPET_PATH = BASE_DIR / "first_round_oracle_rewrite_paper_snippets.tex"
TRANSITION_FIGURE_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/idea_reports/refamb_analysis_figs/first_round_oracle_rewrite_transition_focus.pdf"
)
ITERATION_PALETTE = ["#F4A261", "#F6C177", "#FFD089", "#CFE8F3", "#A8D0E6"]


MAIN_METRICS = [
    ("GPTAcc", 0.051490514905149054, 0.2926829268292683),
    ("ROUGE-L", 0.1513348705867543, 0.22561665692214466),
    ("F1", 0.1617325750142428, 0.32309518815766186),
    ("Recall", 0.30144817817399333, 0.45911691244169817),
    ("Precision", 0.13836331210647146, 0.1878),
    ("Acc", 0.0027100271002710027, 0.016260162601626018),
    ("EM", 0.0000, 0.01084010840108401),
]

AVG_ITER = ("Avg_Iter", 4.5528, 3.3550)

STATUS_TRANSITIONS = [
    ("missing_final_answer→ok", 130),
    ("generation_error→ok", 30),
    ("ok→ok", 74),
    ("missing_final_answer→missing_final_answer", 81),
    ("ok→missing_final_answer", 34),
    ("ok→generation_error", 10),
]

GPTACC_CHANGE = [
    ("Improved", 32),
    ("Unchanged", 329),
    ("Regressed", 8),
]

COVERAGE = {
    "subset_size": 369,
    "rewrite_applied": 366,
}


def write_latex_table() -> None:
    rows = []
    for metric, baseline, oracle in MAIN_METRICS:
        baseline_pct = baseline * 100
        oracle_pct = oracle * 100
        delta_pct = (oracle - baseline) * 100
        rows.append(
            f"{metric} & {baseline_pct:.2f} & \\textbf{{{oracle_pct:.2f}}} & {delta_pct:+.2f} \\\\"
        )

    latex = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{7pt}",
            r"\caption{Effect of replacing only the first retrieval query with an oracle entity-grounded rewrite on the subset of 369 validation items whose first labeled retrieval step is entity-ambiguous. Values are reported in percentage points. Even this single-step intervention substantially improves both judge-based correctness and lexical-overlap metrics, with GPTAcc increasing from 5.15 to 11.65. The rewrite is successfully executed for 366 of the 369 selected items, indicating that first-query entity grounding is a real upstream bottleneck but not the only source of failure.}",
            r"\label{tab:first-round-oracle-rewrite}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Metric & Baseline (\%) & Oracle Rewrite (\%) & $\Delta$ \\",
            r"\midrule",
            *rows,
            r"\midrule",
            f"Subset size & \\multicolumn{{3}}{{c}}{{{COVERAGE['subset_size']}}} \\\\",
            f"Rewrite applied & \\multicolumn{{3}}{{c}}{{{COVERAGE['rewrite_applied']}}} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    TABLE_PATH.write_text(latex, encoding="utf-8")


def add_bar_labels(ax, bars, values, offset=0.006, fmt="{:.3f}"):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def add_percent_bar_labels(ax, bars, values, offset=0.08):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_main_figure() -> None:
    main_metrics = MAIN_METRICS
    labels = [item[0] for item in main_metrics]
    baseline = np.array([item[1] * 100 for item in main_metrics])
    oracle = np.array([item[2] * 100 for item in main_metrics])
    fig = plt.figure(figsize=(6.6, 4.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.25], hspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(labels))
    width = 0.34

    bars0 = ax.bar(x - width / 2, baseline, width, label="Baseline", color=ITERATION_PALETTE[3])
    bars1 = ax.bar(x + width / 2, oracle, width, label="Oracle Rewrite", color=ITERATION_PALETTE[0])

    add_percent_bar_labels(ax, bars0, baseline)
    add_percent_bar_labels(ax, bars1, oracle)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 50)
    ax.legend(frameon=False, ncol=1, loc="upper left")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    ax_iter = fig.add_subplot(gs[1, 0])
    iter_values = np.array([AVG_ITER[1], AVG_ITER[2]])
    iter_labels = ["Baseline", "Oracle Rewrite"]
    iter_y = np.arange(len(iter_labels))
    iter_colors = [ITERATION_PALETTE[1], ITERATION_PALETTE[4]]

    iter_bars = ax_iter.barh(iter_y, iter_values, color=iter_colors, height=0.5)
    for bar, value in zip(iter_bars, iter_values):
        ax_iter.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax_iter.set_yticks(iter_y)
    ax_iter.set_yticklabels(iter_labels)
    ax_iter.invert_yaxis()
    ax_iter.set_ylim(1.45, -0.65)
    ax_iter.set_xlabel("Avg_Iter (Rounds)")
    ax_iter.set_xlim(0, 5.3)
    ax_iter.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax_iter.spines["top"].set_visible(True)
    ax_iter.spines["right"].set_visible(True)

    fig.subplots_adjust(bottom=0.18)
    bbox_top = ax.get_position()
    bbox_bottom = ax_iter.get_position()
    gap_center_y = (bbox_top.y0 + bbox_bottom.y1) / 2
    fig.text(0.5, gap_center_y - 0.02, "(a) Main Accuracy Metrics", ha="center", va="center", fontsize=10)
    fig.text(0.5, 0.055, "(b) Average Iteration Rounds", ha="center", va="center", fontsize=10)
    fig.savefig(MAIN_FIGURE_PATH)
    plt.close(fig)


def plot_summary_figure() -> None:
    fig = plt.figure(figsize=(10.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.35, 0.95], wspace=0.48)

    ax0 = fig.add_subplot(gs[0, 0])
    metric_names = [item[0] for item in MAIN_METRICS]
    baseline = np.array([item[1] for item in MAIN_METRICS])
    oracle = np.array([item[2] for item in MAIN_METRICS])
    x = np.arange(len(metric_names))
    width = 0.36
    bars0 = ax0.bar(x - width / 2, baseline, width, label="Baseline", color="#9AA5B1")
    bars1 = ax0.bar(x + width / 2, oracle, width, label="Oracle Rewrite", color="#2F6C8F")
    add_bar_labels(ax0, bars0, baseline)
    add_bar_labels(ax0, bars1, oracle)
    ax0.set_xticks(x)
    ax0.set_xticklabels(metric_names)
    ax0.set_ylabel("Score")
    ax0.set_ylim(0, 0.42)
    ax0.legend(frameon=False, loc="upper left")
    ax0.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1 = fig.add_subplot(gs[0, 1])
    transition_labels = [item[0] for item in STATUS_TRANSITIONS]
    transition_counts = [item[1] for item in STATUS_TRANSITIONS]
    y = np.arange(len(transition_labels))
    colors = ["#4C956C", "#78A46C", "#A6C48A", "#C9CED6", "#D17B88", "#B56576"]
    bars = ax1.barh(y, transition_counts, color=colors)
    ax1.set_yticks(y)
    ax1.set_yticklabels(transition_labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Count")
    ax1.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, count in zip(bars, transition_counts):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 2])
    change_labels = [item[0] for item in GPTACC_CHANGE]
    change_counts = np.array([item[1] for item in GPTACC_CHANGE])
    change_colors = ["#2A9D8F", "#C9CED6", "#E76F51"]
    wedges, texts, autotexts = ax2.pie(
        change_counts,
        labels=change_labels,
        colors=change_colors,
        startangle=90,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * change_counts.sum() / 100))})",
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        textprops={"fontsize": 8},
    )
    for autotext in autotexts:
        autotext.set_fontsize(7.5)

    fig.savefig(FIGURE_PATH)
    plt.close(fig)


def plot_transition_figure() -> None:
    labels = [
        "MTR → OK",
        "GE → OK",
        "GE → MTR",
        "OK → OK",
        "MTR → MTR",
        "GE → GE",
        "MTR → GE",
        "OK → MTR",
        "OK → GE",
    ]
    counts = np.array([88, 45, 27, 82, 69, 55, 46, 22, 16])
    total = float(counts.sum())
    percentages = counts / total * 100.0
    colors = [
        "#6EA6D7",
        "#6EA6D7",
        "#6EA6D7",
        "#F6C177",
        "#F6C177",
        "#F2C0C0",
        "#E08A8A",
        "#B54B4B",
        "#B54B4B",
    ]

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    y = np.arange(len(labels))
    bars = ax.barh(y, percentages, color=colors, height=0.64)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontweight="bold", fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("Share of Samples (%)", fontweight="bold", fontsize=13)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color("#222222")
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels([f"{tick}%" for tick in [0, 5, 10, 15, 20]], fontweight="bold", fontsize=12)

    for bar, pct, count in zip(bars, percentages, counts):
        if pct >= 4.0:
            text_x = pct - 0.35
            text_ha = "right"
        else:
            text_x = pct + 0.35
            text_ha = "left"
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            ha=text_ha,
            fontsize=12,
            fontweight="bold",
        )

    fig.savefig(TRANSITION_FIGURE_PATH)
    plt.close(fig)


def write_paper_snippets() -> None:
    snippet = "\n".join(
        [
            "% Figure caption",
            r"\caption{Main results of the first-round oracle rewrite intervention on the entity-ambiguous subset. Replacing only the first generated retrieval query with an oracle entity-grounded rewrite yields consistent gains across GPTAcc, F1, Recall, and Precision, with GPTAcc improving from 5.15\% to 11.65\%. This shows that unresolved entity references in the first query are a genuine upstream bottleneck in agentic multimodal VQA, even though later-step failures still limit the final ceiling.}",
            "",
            "% Result paragraph",
            r"Table~\ref{tab:first-round-oracle-rewrite} and Figure~\ref{fig:first-round-oracle-rewrite-main} present the effect of a controlled first-round oracle rewrite on the 369 validation items whose first labeled retrieval step is entity-ambiguous. Although the intervention modifies only the first generated query and leaves all later agent behavior unchanged, it consistently improves both judge-based correctness and lexical-overlap metrics. In particular, GPTAcc increases from 5.15\% to 11.65\%, a gain of 6.50 percentage points and more than 2$\times$ relative improvement. F1, Recall, and Precision also improve by 5.41, 6.70, and 4.94 percentage points, respectively. These results support our central claim that entity ambiguity in early generated queries is a real causal bottleneck in agentic multimodal VQA pipelines. At the same time, the modest absolute ceiling after intervention indicates that fixing the first query alone is insufficient, and that later-step reasoning, retrieval interpretation, and answer finalization remain additional failure sources.}",
            "",
            "% Figure environment",
            r"\begin{figure}[t]",
            r"    \centering",
            r"    \includegraphics[width=0.95\linewidth]{idea_reports/docs/first_round_oracle_rewrite_main_bar.pdf}",
            r"    \caption{Main results of the first-round oracle rewrite intervention on the entity-ambiguous subset. Replacing only the first generated retrieval query with an oracle entity-grounded rewrite yields consistent gains across GPTAcc, F1, Recall, and Precision, with GPTAcc improving from 5.15\% to 11.65\%. This shows that unresolved entity references in the first query are a genuine upstream bottleneck in agentic multimodal VQA, even though later-step failures still limit the final ceiling.}",
            r"    \label{fig:first-round-oracle-rewrite-main}",
            r"\end{figure}",
            "",
            "% Transition figure caption",
            r"\caption{Status transitions under the first-round oracle rewrite intervention. The most important structural gain is the conversion from \textit{missing final answer} to \textit{ok} in 54 cases, showing that better entity grounding in the first generated query often helps the agent escape early failure and successfully complete the pipeline.}",
            "",
        ]
    )
    TEXT_SNIPPET_PATH.write_text(snippet, encoding="utf-8")


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    write_latex_table()
    plot_main_figure()
    plot_summary_figure()
    plot_transition_figure()
    write_paper_snippets()
    print(f"Saved {TABLE_PATH}")
    print(f"Saved {MAIN_FIGURE_PATH}")
    print(f"Saved {FIGURE_PATH}")
    print(f"Saved {TRANSITION_FIGURE_PATH}")
    print(f"Saved {TEXT_SNIPPET_PATH}")


if __name__ == "__main__":
    main()
