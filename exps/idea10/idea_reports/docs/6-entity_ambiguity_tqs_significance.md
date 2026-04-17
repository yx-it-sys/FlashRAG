# Entity Ambiguity and Oracle Rewrite vs TQS: Fairness-Oriented Significance Analysis

## Goal

For fairness, this note reports two separate analyses:

1. On **all samples** in `/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/trajectory_quality_eval/trajectory_quality_samples.jsonl`, compare `TQS` by:
   - whether entity ambiguity is present;
   - the fine-grained entity ambiguity level from `/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/qwen/trajectory_annotation.llm_labeled.adjudicated.jsonl`.
2. On the **369-sample intersection** between:
   - `/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/trajectory_quality_eval/trajectory_quality_samples.jsonl`
   - `/home/you/FlashRAG/exps/idea10/data/result/2026_04_10_12_53_48_first_round_oracle_rewrite_experiment/trajectory_quality_eval/trajectory_quality_samples.jsonl`
   compare `TQS` **before vs after** first-round oracle rewrite using paired significance analysis.

## Grouping Rules

### Sample-level ambiguity indicator

- `entity_ambiguous = true` if any `query_step.llm_label.entity_ambiguous == "Yes"`.
- Otherwise `entity_ambiguous = false`.

### Sample-level ambiguity level

- `No`: no query step is marked entity ambiguous.
- If all ambiguous query steps fall into the same fine-grained category, use that category.
- If a sample contains more than one non-`No` ambiguity category across query steps, mark it as `Mixed`.

Thus the final level set is:

- `No`
- `Object Identification`
- `Indirect Entity Ambiguity`
- `Description`
- `No Object Involved`
- `Mixed`

### Significance procedures

- Unpaired comparisons use:
  - 95% bootstrap confidence intervals for group means and mean differences;
  - permutation tests for mean differences.
- Oracle rewrite comparison on the 369-sample intersection uses:
  - paired bootstrap confidence intervals for the mean paired difference;
  - paired sign-flip permutation tests.

---

## Part I. Full-Sample TQS Comparison by Ambiguity Labels

### Binary Comparison: Ambiguous vs Non-Ambiguous

| Group | Count | Mean TQS | 95% Bootstrap CI |
|-------|------:|---------:|------------------|
| Entity ambiguous | 399 | 0.0973 | [0.0754, 0.1205] |
| Non-entity-ambiguous | 799 | 0.1320 | [0.1111, 0.1538] |

Difference:

- mean difference (`non-ambiguous - ambiguous`) = `0.0347`
- 95% bootstrap CI = `[0.0042, 0.0657]`
- permutation test `p = 0.0448`

### Fine-Grained Comparison by Ambiguity Level

| Level | Count | Mean TQS | 95% Bootstrap CI | Mean Iterations | Mean Total Delta F |
|------|------:|---------:|------------------|----------------:|-------------------:|
| `No` | 799 | 0.1320 | [0.1115, 0.1534] | 2.96 | 0.2282 |
| `Object Identification` | 190 | 0.0891 | [0.0604, 0.1200] | 4.15 | 0.2553 |
| `Indirect Entity Ambiguity` | 68 | 0.1548 | [0.0890, 0.2311] | 3.81 | 0.3505 |
| `Description` | 69 | 0.1441 | [0.0771, 0.2174] | 3.39 | 0.3081 |
| `No Object Involved` | 4 | 0.0000 | [0.0000, 0.0000] | 2.75 | 0.0000 |
| `Mixed` | 68 | 0.0211 | [0.0059, 0.0417] | 4.93 | 0.1176 |

### Fine-Grained Difference vs `No`

Here the reported difference is `mean(No) - mean(Level)`.

| Level | Mean Diff vs `No` | 95% Bootstrap CI | Permutation p-value |
|------|-------------------:|------------------|--------------------:|
| `Object Identification` | 0.0429 | [0.0064, 0.0787] | 0.0668 |
| `Indirect Entity Ambiguity` | -0.0228 | [-0.0975, 0.0484] | 0.5547 |
| `Description` | -0.0121 | [-0.0899, 0.0569] | 0.7479 |
| `No Object Involved` | 0.1320 | [0.1122, 0.1534] | 0.5275 |
| `Mixed` | 0.1109 | [0.0828, 0.1376] | 0.0038 |

### Part I Findings

1. At the binary level, entity ambiguity is still associated with lower `TQS`.

- Ambiguous samples average `0.0973`, versus `0.1320` for non-ambiguous samples.
- The gap is `0.0347`, with 95% bootstrap CI `[0.0042, 0.0657]`.
- The permutation test gives `p = 0.0448`.

2. The fine-grained pattern is heterogeneous rather than monotonic.

- `Object Identification` has lower mean `TQS` than `No`.
- `Mixed` is much lower than `No`, and this is the clearest fine-grained contrast.
- `Indirect Entity Ambiguity` and `Description` are not lower than `No` under the current metric.

3. The most severe degradation is concentrated in `Mixed` cases.

- `Mixed` has the lowest mean `TQS` (`0.0211`), the highest mean iteration count (`4.93`), and a small mean `total_delta_f` (`0.1176`).
- This suggests that trajectories with multiple ambiguity modes are the hardest cases under the current evaluation setup.

4. `No Object Involved` is too small to support strong inference.

- It contains only 4 samples, so its mean and CI are not stable enough for interpretation.

### Mixed Composition Breakdown

`Mixed` means that a sample contains more than one non-`No` ambiguity category across labeled query steps. In the current adjudicated label file, the `68` `Mixed` samples break down into the following combinations:

| Mixed combination | Count | Example annotation IDs |
|---|---:|---|
| `Description + Object Identification` | 30 | `a9b3ce90-e15e-4161-ad6b-2b189df009e6`, `eda47d0d-50c8-4a57-9b9c-818527e842a1`, `cccf641a-285a-4d6f-91ce-f4cd34cd8a5a`, `c050265c-8428-4dee-8cfd-7706a8066e64`, `b997879c-0cd2-49c1-8a43-6e50456f9fe2` |
| `Indirect Entity Ambiguity + Object Identification` | 17 | `b58a53ef-fd49-40c2-b9b6-f353539749ae`, `0a33f53f-719b-48a4-8b76-5e0f78c528ff`, `08029d66-61cb-4c23-b762-1ee7648608f2`, `c5ec19f2-800c-444d-9284-529dbe0baf8a`, `9a39e0e6-3625-47f9-b1c7-c213f7eda2e8` |
| `Description + Indirect Entity Ambiguity` | 15 | `1d9ab7e3-3599-464c-971c-123ac94a5e91`, `38374032-9e4c-4594-8e6b-74d6e07a780a`, `132f522d-5cac-4c39-813c-70d2b6ffa6ee`, `b0790de5-87fd-4233-8103-dc2f17f0a2e6`, `ea6a8cb3-f336-461c-a1f7-98d8735e686d` |
| `Description + Indirect Entity Ambiguity + Object Identification` | 3 | `e0967c91-77aa-4c6a-8dcb-197d6718174e`, `f57b9082-a2de-47fd-90b0-1b203c6b1d62`, `53f98721-bbf1-4dd0-b763-f4140a55f9e5` |
| `Indirect Entity Ambiguity + No Object Involved + Object Identification` | 1 | `2be02caa-d5d1-48ff-a29c-2e2e67dbdcdc` |
| `No Object Involved + Object Identification` | 1 | `2f81e89c-d011-4957-b925-73d8c4bf0ac8` |
| `Description + Indirect Entity Ambiguity + No Object Involved + Object Identification` | 1 | `57147f41-23f4-4e0e-8ed3-f173dcc258f1` |

The dominant `Mixed` patterns are therefore pairwise combinations centered on `Object Identification`, especially `Description + Object Identification` and `Indirect Entity Ambiguity + Object Identification`.

---

### Paired Comparison Stratified by Baseline Status

### Paired Overall Comparison

| Setting | Count | Mean TQS | 95% Bootstrap CI |
|--------|------:|---------:|------------------|
| Baseline | 369 | 0.1035 | [0.0795, 0.1292] |
| Oracle rewrite | 369 | 0.1532 | [0.1253, 0.1818] |

Paired difference:

- mean difference (`oracle - baseline`) = `0.0497`
- 95% paired bootstrap CI = `[0.0122, 0.0874]`
- paired sign-flip permutation `p = 0.0072`
- Wilcoxon signed-rank `W = 6580.5`, `p = 0.001021`
- matched-pairs rank-biserial correlation = `0.2747`

Outcome counts on the 369-sample intersection:

- improved: `119`
- unchanged: `179`
- regressed: `71`

### Paired Comparison Stratified by Baseline Status

| Baseline Status | Count | Baseline Mean TQS | Oracle Mean TQS | Mean Diff (`oracle - baseline`) | 95% Paired Bootstrap CI | Paired p-value | Wilcoxon $W$ | Wilcoxon p-value | Rank-biserial |
|----------------|------:|------------------:|----------------:|--------------------------------:|-------------------------|----------------:|-------------:|-----------------:|--------------:|
| `generation_error` | 50 | 0.0258 | 0.1684 | 0.1426 | [0.0739, 0.2222] | 0.0002 | 34.5 | 0.000205 | 0.8175 |
| `missing_final_answer` | 211 | 0.0169 | 0.1625 | 0.1456 | [0.1087, 0.1844] | 0.0002 | 413.0 | 5.989e-13 | 0.8331 |
| `ok` | 108 | 0.3086 | 0.1281 | -0.1806 | [-0.2646, -0.0980] | 0.0004 | 438.0 | 5.55e-05 | -0.5788 |

### Part II Findings

1. On the fair 369-sample intersection, oracle rewrite significantly improves mean `TQS` overall.

- Mean `TQS` rises from `0.1035` to `0.1532`.
- The paired gain is `0.0497`, with 95% CI `[0.0122, 0.0874]`.
- The paired test gives `p = 0.0072`.

2. The gain is driven by samples that were failing before intervention.

- For baseline `generation_error`, the mean gain is `0.1426`.
- For baseline `missing_final_answer`, the mean gain is `0.1456`.
- Both effects are strongly significant.

3. For samples that were already `ok` under the baseline, `TQS` drops after oracle rewrite.

- In the baseline-`ok` subset, mean `TQS` changes from `0.3086` to `0.1281`.
- The paired difference is `-0.1806`, with CI entirely below zero.

4. Therefore, oracle rewrite helps mainly by rescuing low-quality failing trajectories, not by uniformly improving already-successful ones.

---

## Overall Takeaway

The fairness-oriented reading should separate the two questions clearly.

### A. Does entity ambiguity correlate with lower trajectory quality on the full dataset?

Yes, at the coarse binary level.

- Ambiguous samples have lower mean `TQS` than non-ambiguous samples.
- But the fine-grained picture is not uniform: the biggest penalty is concentrated in `Mixed` and, more weakly, `Object Identification` cases.

### B. Does first-round oracle rewrite improve trajectory quality when evaluated on the same sample set before and after intervention?

Yes, on the 369-sample intersection.

- The paired overall effect is positive and significant.
- The improvement comes primarily from cases that were failing under the baseline.
- Already-successful baseline cases show a decrease in `TQS` after intervention.

So the fairest conclusion is:

- entity ambiguity is associated with lower `TQS` overall, but the effect is concentrated in specific ambiguity regimes rather than uniformly present across all fine-grained levels;
- first-round oracle rewrite significantly improves `TQS` on the matched subset overall, chiefly by repairing trajectories that otherwise end in failure.
