# Cross-Source Consistency Analysis Within Task Type

## Scope
- Task types: `Single-hop Attribute Query`, `Entity Recognition`.
- Sources: `crag`, `infoseek`, `mcsearch`, `oven` (only available sources per task are tested).
- Model-run metrics from `RefAmb_original_Qwen2.5-vl-7B`:
  - `RTE`: `trajectory_quality_score` from `trajectory_quality_eval_whole_delta_F_updated/trajectory_quality_samples.jsonl`.
  - `LJ`: `gpt_acc` from `gpt_acc_score.json`.
  - `OK State Ratio`: proportion with `status == ok` in trajectory-quality samples.
- Data-property metrics from `origin.jsonl`: question length, answer length, `text_alone_identifies_entity`, `external_evidence_dependency`.

## Equivalence Test Setup (TOST, alpha=0.05)
- `question_len` margin: ±3.0.
- `answer_len` margin: ±5.0.
- `text_alone_yes_ratio` margin: ±0.1.
- `external_dep_yes_ratio` margin: ±0.1.
- `rte` margin: ±0.05.
- `lj` margin: ±0.05.
- `ok_ratio` margin: ±0.1.

## Sample Coverage
| Task Type | Source | N items |
|---|---:|---:|
| Entity Recognition | crag | 83 |
| Entity Recognition | infoseek | 11 |
| Entity Recognition | oven | 206 |
| Single-hop Attribute Query | crag | 511 |
| Single-hop Attribute Query | infoseek | 581 |

## Descriptive Statistics (mean/proportion by source)
| Task | Source | Q len | A len | Text-alone Yes | External-dep Yes | RTE | LJ | OK ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Entity Recognition | crag | 6.41 | 12.08 | 0.072 | 1.000 | 0.320 | 0.301 | 0.554 |
| Entity Recognition | infoseek | 7.00 | 1.73 | 0.000 | 1.000 | 0.224 | 0.182 | 0.273 |
| Entity Recognition | oven | 6.74 | 1.26 | 0.000 | 0.515 | 0.162 | 0.422 | 0.699 |
| Single-hop Attribute Query | crag | 7.43 | 14.23 | 0.080 | 1.000 | 0.402 | 0.222 | 0.108 |
| Single-hop Attribute Query | infoseek | 8.99 | 2.01 | 0.000 | 1.000 | 0.164 | 0.055 | 0.077 |

## Pairwise TOST Results Summary
### Single-hop Attribute Query
| Metric | Equivalent pairs / Total pairs | Equivalent pairs | Non-equivalent pairs |
|---|---:|---|---|
| answer_len | 0/1 | - | crag-infoseek |
| external_dep_yes_ratio | 1/1 | crag-infoseek | - |
| lj | 0/1 | - | crag-infoseek |
| ok_ratio | 1/1 | crag-infoseek | - |
| question_len | 1/1 | crag-infoseek | - |
| rte | 0/1 | - | crag-infoseek |
| text_alone_yes_ratio | 0/1 | - | crag-infoseek |

### Entity Recognition
| Metric | Equivalent pairs / Total pairs | Equivalent pairs | Non-equivalent pairs |
|---|---:|---|---|
| answer_len | 1/3 | infoseek-oven | crag-infoseek, crag-oven |
| external_dep_yes_ratio | 1/3 | crag-infoseek | crag-oven, infoseek-oven |
| lj | 0/3 | - | crag-infoseek, crag-oven, infoseek-oven |
| ok_ratio | 0/3 | - | crag-infoseek, crag-oven, infoseek-oven |
| question_len | 3/3 | crag-infoseek, crag-oven, infoseek-oven | - |
| rte | 0/3 | - | crag-infoseek, crag-oven, infoseek-oven |
| text_alone_yes_ratio | 1/3 | infoseek-oven | crag-infoseek, crag-oven |

## Interpretation
- This analysis checks whether cross-source differences are practically small under pre-defined equivalence margins, instead of only testing for difference.
- If a metric shows many non-equivalent pairs, cross-source heterogeneity remains and should be controlled or stratified in downstream claims.
- If most pairs are equivalent (especially for RTE, LJ, and OK ratio), it supports the claim that within-task cross-source inconsistency is limited.

## Output Files
- Summary stats: `/home/you/FlashRAG/exps/idea10/idea_reports/new/reports/cross_source_consistency_summary_by_tasktype_source.csv`
- Pairwise TOST: `/home/you/FlashRAG/exps/idea10/idea_reports/new/reports/cross_source_consistency_tost_pairwise.csv`
