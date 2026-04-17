# Sampling Plan for Reproducing OmniSearch on a Larger API Model

## Goal

This note defines a concrete sampling plan for reproducing the OmniSearch pipeline on a larger API-based multimodal model, with the specific goal of testing whether entity ambiguity remains a major bottleneck when model scale increases.

The plan is designed to answer two different questions at the same time:

1. In the overall validation distribution, how often does entity ambiguity still appear on the larger model, and how strongly is it associated with failure?
2. On the subset that the smaller model already found difficult due to entity ambiguity, how often can the larger model repair the problem?

Because these are different questions, the experiment should not rely on a single sample pool.

---

## Principle

Use a two-set design:

1. A population-oriented stratified sample from the full validation set.
2. An ambiguity-focused stratified sample from the smaller-model entity-ambiguous subset.

Do not only sample from the ambiguous subset. That would overestimate the frequency of ambiguity on the larger model and would not support a population-level conclusion.

Do not only sample from the full dataset. That would make the hard ambiguity cases too sparse for detailed repair analysis.

---

## Source Files

### Full validation universe

- Dataset split used by OmniSearch:
  - `/home/you/FlashRAG/exps/idea10/data/datasets/crag_mm/validation.jsonl`

### Smaller-model baseline trajectory outputs

- Baseline trajectories:
  - `/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/omnisearch_trajectories.jsonl`
- Baseline entity ambiguity labels:
  - `/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/qwen/trajectory_annotation.llm_labeled.adjudicated.jsonl`

### First-round oracle rewrite reference subset

- Rewrite trajectories:
  - `/home/you/FlashRAG/exps/idea10/data/result/2026_04_10_12_53_48_first_round_oracle_rewrite_experiment/omnisearch_trajectories.jsonl`
- Rewrite-side entity ambiguity labels:
  - `/home/you/FlashRAG/exps/idea10/data/result/2026_04_10_12_53_48_first_round_oracle_rewrite_experiment/label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.jsonl`

For the new large-model reproduction experiment, the sampling frame should be defined using the smaller-model baseline labels, not using the oracle-rewrite run.

---

## Recommended Sample Budget

Use `200` total items.

Split the budget into:

1. `Set A: Population sample` = `120` items
2. `Set B: Ambiguity-focused sample` = `80` items

This budget is large enough to support:

- a reasonable estimate of ambiguity frequency on the larger model;
- a targeted analysis of whether ambiguity-specific failure modes are repaired;
- a manageable API cost for end-to-end OmniSearch reproduction.

If the API budget is tighter, use `80` total items with the same structure:

- `Set A = 50`
- `Set B = 30`

If budget is tighter, a fallback is `120` total items:

- `Set A = 80`
- `Set B = 40`

The `200`-item design is the default recommendation.

---

## Set A: Population-Oriented Stratified Sample

### Purpose

Estimate what happens on the larger model under the real validation distribution.

This set is the only set that should be used to report:

- large-model entity ambiguity frequency;
- large-model answer quality under the natural distribution;
- whether ambiguity remains predictive of failure overall.

### Sampling Frame

All validation items that have:

- a valid baseline OmniSearch trajectory;
- a matching baseline ambiguity label record.

### Primary Stratification Variables

Use the following stratification order:

1. `baseline item-level ambiguity`
   - `ambiguous`
   - `non-ambiguous`
2. `baseline status`
   - `ok`
   - `missing_final_answer`
   - `generation_error` or other failure
3. `baseline trajectory length`
   - `short`: 1 retrieval-result turn
   - `medium`: 2 to 3 retrieval-result turns
   - `long`: 4+ retrieval-result turns

### Allocation Rule

Sample `120` items approximately in proportion to the natural distribution, while enforcing minimum coverage of rare but important strata.

Recommended allocation:

| Stratum | Target Count |
|---|---:|
| Non-ambiguous items | 78 |
| Ambiguous items | 42 |

Within each ambiguity group, allocate across status and length using proportional sampling with these floors:

- at least `10` items from each status bucket that exists;
- at least `8` items from each length bucket that exists;
- if a bucket is too small, take all available items and redistribute the remainder proportionally.

### Why this design

The full validation set is dominated by non-ambiguous items. Preserving that imbalance is necessary if the goal is to estimate the larger model's true ambiguity rate. The small floor constraints prevent the sample from collapsing into only easy `ok` cases.

---

## Set B: Ambiguity-Focused Stratified Sample

### Purpose

Measure how often the larger model repairs cases that the smaller model found entity-ambiguous.

This set is the only set that should be used to report:

- ambiguity repair rate;
- ambiguity persistence rate;
- which ambiguity subtypes remain difficult after scaling up the generator.

### Sampling Frame

Items that are labeled as entity-ambiguous at the item level under the smaller-model adjudicated label file.

That is:

- `item-level ambiguous = true` if any `query_step.llm_label.entity_ambiguous == "Yes"` in the adjudicated baseline label file.

### Primary Stratification Variables

Use the following stratification order:

1. `sample-level ambiguity level`
   - `Object Identification`
   - `Indirect Entity Ambiguity`
   - `Description`
   - `Mixed`
   - `No Object Involved` if enough samples exist; otherwise include all available and do not treat as a major stratum
2. `baseline status`
   - `ok`
   - `missing_final_answer`
   - `generation_error` or other failure
3. `trajectory length`
   - `short`
   - `medium`
   - `long`

### Sample-Level Ambiguity Level Definition

Use the same rule as in the current significance note:

- `No` if no step is ambiguous
- otherwise if all ambiguous steps share one category, use that category
- if an item contains more than one ambiguous category, mark it as `Mixed`

### Allocation Rule

Sample `80` items with the following target allocation:

| Ambiguity Level | Target Count |
|---|---:|
| Object Identification | 28 |
| Indirect Entity Ambiguity | 18 |
| Description | 16 |
| Mixed | 16 |
| No Object Involved | up to 2, only if available |

Adjustment rule:

- If `No Object Involved` has fewer than 2 available items, include all available and reallocate the remainder to `Mixed`.
- Within each ambiguity level, split items across baseline status with priority toward failure cases:
  - about `50%` from `missing_final_answer` / failure
  - about `25%` from `ok`
  - about `25%` from other available status buckets
- Within each status subset, preserve some spread over `short`, `medium`, and `long` trajectories.

### Why this design

This set deliberately over-samples the ambiguity regimes that matter most for mechanism analysis. It is not intended for estimating the overall large-model ambiguity frequency.

---

## Overlap Rule Between Set A and Set B

Use `disjoint` sampling.

An item selected into `Set A` must not appear in `Set B`.

Reason:

- `Set A` is for unbiased population-style reporting.
- `Set B` is for stress-testing ambiguity repair.

Keeping them disjoint avoids confusion in downstream analysis and keeps weighting simple.

---

## What to Measure on the Larger Model

For each sampled item, run the full OmniSearch pipeline on the larger API generator and then annotate the resulting trajectories with the same entity ambiguity protocol.

### Item-level metrics

- whether the reproduced large-model trajectory contains any entity-ambiguous step;
- final status: `ok`, `missing_final_answer`, `generation_error`, etc.;
- final answer correctness metrics already used in the project;
- total number of retrieval rounds.

### Step-level metrics

- ambiguous step count;
- ambiguous step rate among text retrieval steps;
- fine-grained ambiguity category counts.

### Transition metrics relative to the smaller model

For each sampled item, compare smaller-model baseline vs larger-model reproduction:

- `ambiguous -> ambiguous`
- `ambiguous -> non-ambiguous`
- `non-ambiguous -> ambiguous`
- `non-ambiguous -> non-ambiguous`

The key quantity for `Set B` is:

- `repair rate = ambiguous -> non-ambiguous / baseline ambiguous`

The key quantity for `Set A` is:

- large-model item-level ambiguity rate under the natural distribution.

---

## Reporting Rules

### Report Set A and Set B separately

Never merge them into one overall frequency estimate.

Correct reporting pattern:

1. `Set A`:
   - "On a population-oriented stratified sample, the large model shows an item-level ambiguity rate of X and a step-level ambiguity rate of Y."
2. `Set B`:
   - "On the smaller-model ambiguity subset, the large model resolves Z% of previously ambiguous items."

### Core tables to include

1. `Set A` overall before-vs-after ambiguity frequency table
2. `Set B` ambiguity repair table by ambiguity level
3. `Set B` ambiguity repair table by baseline status
4. Optional qualitative case table with:
   - repaired examples
   - persistent ambiguity examples
   - regressions

---

## Recommended Export Artifacts

Create the following files for reproducibility:

1. `sample_set_A_population.jsonl`
2. `sample_set_B_ambiguity_focused.jsonl`
3. `sample_manifest.csv`

The manifest should contain at least:

- `id`
- `set_name`
- `baseline_item_ambiguous`
- `baseline_item_level`
- `baseline_status`
- `baseline_num_retrieval_turns`
- `baseline_num_ambiguous_steps`
- `sampling_weight`

For `Set A`, use sampling weights if any stratum had to be over- or under-sampled relative to the natural distribution.
For `Set B`, weights are not needed because it is an analysis-only stress set.

---

## Concrete Execution Recommendation

Use the following exact plan.

### Main plan

- Total: `200` items
- `Set A`: `120` items, population-oriented, ambiguity/status/length stratified
- `Set B`: `80` items, ambiguity-focused, level/status/length stratified

### If only one set can be run first

Run `Set B` first.

Reason:

- the research question is specifically about whether model scaling reduces the impact of entity ambiguity;
- `Set B` gives the clearest signal about repair on the hard cases;
- it is the fastest way to tell whether the hypothesis is promising.

Then run `Set A` as the validation set for generalization.

---

## Decision Rule for the Next Step

After the first `60` to `80` large-model reproductions, decide whether to scale up based on these thresholds:

1. If the large model repairs fewer than `20%` of baseline ambiguous items in `Set B`, entity ambiguity is probably not substantially alleviated by scale alone.
2. If the large model repairs at least `40%` of baseline ambiguous items in `Set B`, continue to full `Set A + Set B` evaluation.
3. If the large model substantially lowers ambiguity but does not improve final answer quality, then ambiguity is only one bottleneck and later-turn control should become the next focus.

---

## Final Recommendation

The recommended design is:

- use `two disjoint stratified samples`;
- preserve the natural ambiguity prevalence in `Set A`;
- over-sample ambiguity cases in `Set B`;
- report population effects and repair effects separately.

This gives a defensible answer to both:

- whether entity ambiguity is still common at larger scale;
- whether larger models actually resolve the ambiguity cases that currently hurt OmniSearch.
