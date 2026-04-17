# Metric Effectiveness Experiment Guidelines

## Goal

This document specifies how to validate that the proposed trajectory-quality metric is not only computable, but also meaningful, discriminative, and useful for analyzing multi-step retrieval trajectories.

The target metric currently includes:

- step-level information increment `Delta F_t`;
- step-level iteration utility `u_t`;
- step-level score `S_t = Delta F_t * u_t`;
- trajectory-level score `TQS = average_t S_t`.

The main claim to defend is:

- the metric measures whether each retrieval step contributes genuinely new and answer-relevant information, and therefore provides a better signal of trajectory quality than simple proxies such as answer correctness, trajectory length, or repeated-query statistics.


## Core Validation Questions

The experiment section should answer the following questions:

1. Does the metric align with human judgments of trajectory quality?
2. Does the metric outperform simpler baselines and proxy metrics?
3. Does the metric respond correctly when a trajectory is intentionally degraded?
4. Does the metric correlate with task success without collapsing into an answer-correctness proxy?
5. Is the metric robust to judge noise and implementation choices?


## Recommended Validation Story

The paper should revolve around one primary claim and one supporting claim.

- Primary claim:
  - `TQS` captures meaningful trajectory progress better than naive success or length-based metrics.
- Supporting claim:
  - the two components, `Delta F_t` and `u_t`, are both necessary: novelty alone is insufficient, and usefulness alone is insufficient.

Anti-claims to rule out:

- the metric is only measuring trajectory length;
- the metric is only reproducing final answer correctness;
- the metric is unstable because it depends on LLM judging;
- the metric only works because of a specific threshold or prompt.


## Experiment 1: Human Alignment

### Purpose

Verify that the metric reflects the intended notion of trajectory quality rather than an arbitrary numerical artifact.

### Data

- Sample `150-300` trajectories from `trajectory_quality_samples.jsonl`.
- Use stratified sampling to cover:
  - high / medium / low `trajectory_quality_score`;
  - `status = ok` and `status = missing_final_answer`;
  - short, medium, and long trajectories;
  - text retrieval, image retrieval, and no retrieval cases.

### Human Annotation Protocol

For each step, annotators should label:

- whether the step introduces new information;
- whether the step is useful for answering the question;
- whether the step is redundant or repeated;
- whether the step should be considered high-quality, medium-quality, or low-quality.

For each full trajectory, annotators should label:

- overall trajectory quality: `high / medium / low`;
- whether the trajectory makes meaningful progress;
- whether the trajectory is efficient;
- whether the trajectory contains obvious wasted steps.

### What to Compare

At step level:

- `Delta F_t` vs human novelty labels;
- `u_t` vs human usefulness labels;
- `S_t` vs human step-quality labels.

At trajectory level:

- `TQS` vs human overall trajectory-quality labels;
- `TQS` vs pairwise human preference between two trajectories for the same question.

### Metrics

Use:

- Spearman correlation;
- Kendall tau;
- AUC for high-vs-low quality discrimination;
- pairwise ranking accuracy;
- Cohen's kappa or Krippendorff's alpha for inter-annotator agreement.

### Success Criterion

The metric is credible if:

- `Delta F_t` aligns with perceived novelty;
- `u_t` aligns with perceived usefulness;
- `TQS` is positively and significantly correlated with human-rated trajectory quality.


## Experiment 2: Comparison Against Simpler Baselines

### Purpose

Demonstrate that the proposed metric is not simply reproducing obvious heuristics.

### Baselines

Compare against:

- final answer correctness;
- binary trajectory completion status;
- trajectory length `num_iterations`;
- number of unique queries;
- mean query similarity;
- repeated-query ratio;
- utility-only score;
- novelty-only score;
- raw accumulated fact count;
- number of retrieval steps with non-empty extracted facts.

### Evaluation Targets

Compare each metric against:

- human trajectory-quality labels;
- human step-quality labels;
- final answer correctness;
- pairwise human preference between trajectories.

### What to Report

For each metric, report:

- correlation with human trajectory quality;
- AUC for high-vs-low human quality;
- pairwise ranking accuracy;
- partial correlation after controlling for `num_iterations`;
- regression improvement when adding `TQS` on top of simple baselines.

### Stronger Analysis

Run simple regression models:

- baseline-only predictors;
- baseline predictors plus `TQS`;
- baseline predictors plus `Delta F`;
- baseline predictors plus `u_t`.

If adding `TQS` materially improves fit or ranking accuracy, this is strong evidence that the metric provides information beyond simple proxies.


## Experiment 3: Intervention Sensitivity

### Purpose

Show that the metric responds in the correct direction when trajectory quality is intentionally degraded.

### Construction of Degraded Trajectories

Starting from original trajectories, construct controlled variants with one perturbation at a time:

- query repetition:
  - replace later queries with near-duplicates of earlier queries;
- irrelevant evidence injection:
  - replace a useful retrieval result with irrelevant evidence;
- critical-step deletion:
  - remove a step that contributes a key accepted fact;
- filler-step insertion:
  - insert one or two clearly redundant steps;
- order corruption:
  - swap steps whose order matters for progress.

### Expected Metric Response

- `Delta F_t` should decrease when genuinely new facts are removed;
- `u_t` should decrease when useful steps are replaced by irrelevant ones;
- `TQS` should decrease for degraded trajectories in aggregate.

### What to Report

Report:

- original vs degraded metric distributions;
- paired statistical tests;
- mean score drop per perturbation type;
- examples where the metric succeeds and fails.

This experiment is especially important because it tests whether the metric is sensitive to causal quality changes rather than only descriptive correlations.


## Experiment 4: Task-Level Validity

### Purpose

Show that the metric has system-level meaning, while remaining more informative than final answer correctness alone.

### Analyses

Run the following analyses:

- `TQS` vs final answer correctness;
- `TQS` vs `status`;
- bucket trajectories into low / medium / high `TQS` and compare answer accuracy;
- within correct-answer trajectories only, compare high- and low-`TQS` cases;
- within failed trajectories only, compare high- and low-`TQS` cases;
- control for `num_iterations` and question difficulty.

### Interpretation

Desired findings:

- higher `TQS` should generally correspond to better final outcomes;
- among successful trajectories, the metric should still separate efficient and evidence-grounded solutions from lucky or weak ones;
- among failed trajectories, the metric should separate near-miss trajectories from clearly unproductive ones.

If the metric only mirrors final correctness and adds no internal discrimination, the claim is weak.


## Experiment 5: Robustness and Ablation

### Purpose

Verify that the metric is stable enough to be trusted and that each design choice is justified.

### Required Ablations

Run at least:

- `Delta F` only;
- `u_t` only;
- remove query-similarity gating;
- change `sim_threshold`;
- change `fact_sim_threshold`;
- replace trajectory averaging with alternative aggregation rules;
- vary fact extraction prompt;
- vary utility-labeling prompt;
- repeat LLM judge runs on a subset to estimate judge variance.

### Key Questions

- Does the metric remain correlated with human labels under moderate threshold changes?
- Is the full metric better than either component alone?
- Does similarity gating actually suppress repeated-query false gains?
- Is the metric robust enough that conclusions do not disappear under small implementation changes?

### What to Report

Report:

- main metric performance;
- ablated metric performance;
- standard deviation across repeated judge runs;
- sensitivity curves for threshold sweeps.


## Immediate Analyses from Current Results

Before launching new annotation or intervention experiments, run descriptive analyses on:

- `/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/trajectory_quality_eval/trajectory_quality_samples.jsonl`

Generate the following plots and summary tables first:

- histogram of `trajectory_quality_score`;
- histogram of `total_delta_f`;
- box plot of `trajectory_quality_score` by `status`;
- scatter plot of `trajectory_quality_score` vs `num_iterations`;
- scatter plot of `trajectory_quality_score` vs `total_delta_f`;
- distribution of step scores by retrieval type;
- percentage of zero-score trajectories;
- fraction of repeated-query gated steps;
- summary statistics for `status = ok` vs `status = missing_final_answer`.

These descriptive analyses are useful for checking whether the metric has a healthy distribution or whether it collapses to trivial patterns such as:

- too many trajectories with zero score;
- overly strong correlation with length;
- no separation between successful and failed trajectories.


## Suggested Table and Figure Plan

Use a compact paper-oriented presentation.

### Main Paper

- Table 1:
  - human alignment results for `Delta F_t`, `u_t`, and `TQS`;
- Table 2:
  - comparison against baseline metrics;
- Figure 1:
  - score distributions for successful vs failed trajectories;
- Figure 2:
  - score drop under controlled trajectory degradation;
- Table 3:
  - ablation and robustness results.

### Appendix

- annotation guidelines;
- inter-annotator agreement table;
- additional threshold sweeps;
- prompt variants;
- more qualitative trajectory case studies.


## Recommended Run Order

### Stage 0: Sanity Check

- verify result schema;
- compute score distributions;
- check for pathological collapse or length bias.

Go / no-go decision:

- proceed only if the metric distribution is not degenerate.

### Stage 1: Human Annotation

- sample and annotate `150-300` trajectories;
- compute inter-annotator agreement;
- finalize human-label definitions.

Go / no-go decision:

- proceed only if annotation quality is acceptable and the labels are usable.

### Stage 2: Baseline Comparison

- compute all simple proxy metrics;
- compare against human labels and success labels.

### Stage 3: Intervention Sensitivity

- generate degraded trajectories;
- rerun metric;
- analyze paired score changes.

### Stage 4: Robustness and Ablation

- threshold sweeps;
- component removal;
- repeated judge runs.


## Biggest Risks

The most important risks to address explicitly are:

- LLM judge noise in utility labeling;
- instability caused by fact extraction quality;
- excessive dependence on semantic similarity thresholds;
- length bias from trajectory-level averaging;
- overclaiming from correlation with final correctness.

Each of these should appear in the experimental discussion, not only in internal notes.


## Final Checklist

- [ ] Human annotation protocol is defined and executable.
- [ ] Human alignment results are reported at both step and trajectory level.
- [ ] Strong baseline metrics are included.
- [ ] Intervention-based validation is included.
- [ ] Task-level validity is shown without collapsing into correctness-only analysis.
- [ ] Robustness and ablations are reported.
- [ ] Distribution sanity checks are included.
- [ ] Main paper figures and tables are mapped before running large-scale experiments.
