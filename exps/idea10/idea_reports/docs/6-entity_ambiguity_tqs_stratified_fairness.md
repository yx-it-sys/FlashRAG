# Stratified Fairness Analysis for Entity Ambiguity vs TQS

Stratification variables:
- `status ∈ {ok, missing_final_answer, generation_error}`
- `num_iterations bucket ∈ {1-2, 3-4, 5+}`

Within each stratum, the reported difference is `mean(non-ambiguous) - mean(ambiguous)`.

| Status | Iter Bucket | Ambiguous n | Non-ambiguous n | Mean TQS (Amb) | Mean TQS (Non) | Diff | 95% Bootstrap CI | Permutation p |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `generation_error` | `3-4` | 1 | 5 | 0.1250 | 0.0000 | -0.1250 | [-0.1250, -0.1250] | 0.1502 |
| `generation_error` | `5+` | 57 | 95 | 0.0222 | 0.0118 | -0.0104 | [-0.0278, 0.0047] | 0.1722 |
| `missing_final_answer` | `3-4` | 1 | 2 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| `missing_final_answer` | `5+` | 231 | 253 | 0.0167 | 0.0174 | 0.0007 | [-0.0079, 0.0092] | 0.8756 |
| `ok` | `1-2` | 94 | 428 | 0.3285 | 0.2314 | -0.0971 | [-0.1852, -0.0109] | 0.0244 |
| `ok` | `3-4` | 14 | 15 | 0.1756 | 0.0611 | -0.1145 | [-0.2262, 0.0052] | 0.0782 |

## Weighted Summary

- Number of valid strata with both groups present: `6`
- Size-weighted average within-stratum TQS difference (`non-ambiguous - ambiguous`): `-0.0468`
