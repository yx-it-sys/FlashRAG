# Human Alignment Consistency Analysis

- Spearman correlation between trajectory-level human overall labels (mapped to `3/2/1`) and `TQS`: `rho = 0.6382` (`p = 1.585e-18`).
- Kendall tau between trajectory-level human overall labels (mapped to `3/2/1`) and `TQS`: `tau = 0.5842` (`p = 1.585e-18`).

# Fined-grained Consistency Analysis
## mean_delta_f vs mean_h_novelty
  Spearman rho = 0.6001 (p = 7.577e-16)
  Kendall tau = 0.5419 (p = 7.95e-14)
## mean_utility vs mean_h_usefulness
  Spearman rho = 0.8467 (p = 7.491e-42)
  Kendall tau = 0.8009 (p = 1.952e-28)
## TQS vs human_proxy_tqs
  Spearman rho = 0.7508 (p = 4.404e-28)
  Kendall tau = 0.7031 (p = 3.605e-22)
## mean_step_score vs human_proxy_tqs
  Spearman rho = 0.7508 (p = 4.404e-28)
  Kendall tau = 0.7031 (p = 3.605e-22)

## Few-wasted steps vs Delta F+avg
Human `wasted_steps` labels are mapped to `3/2/1`:
`No obvious wasted steps -> 3`, `Some wasted steps -> 2`, `Many wasted steps -> 1`.
`Delta F+avg` is computed as the average `delta_f` over steps with `delta_f > 0`.
Valid sample count: `n = 148` (`2` samples skipped because the metric file has no steps).

  Spearman rho = 0.5349 (p = 2.496e-12)
  Kendall tau = 0.5067 (p = 7.565e-11)

## Few-wasted steps vs non-wasted_ratio
`non-wasted_ratio` is computed as the fraction of steps with `step_score > 0`.

  Spearman rho = 0.6140 (p = 1.053e-16)
  Kendall tau = 0.5863 (p = 7.680e-14)
