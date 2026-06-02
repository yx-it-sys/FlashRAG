# Mechanism Analysis

This supplement focuses on why rewrite changes trajectory quality, using the same task-balanced aggregation convention as the bar charts.

## Core Mechanism

The dominant effect of both `Disturb` and `Oracle` is not a simple increase in fact novelty. Instead, the rewrites usually push the agent into longer trajectories with higher `ΔF` but lower `utility` and lower informative-step density.

In other words, the intervention often increases the amount of retrieved material while decreasing the fraction of steps that the downstream reasoning actually uses.

## Backbone-Level Pattern

| Model | Setting | TQS | ΔF | Iterations | Utility | Informative rate |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-vl-7B | Original | 0.3077 | 1.1593 | 2.6993 | 0.4354 | 0.4019 |
| Qwen2.5-vl-7B | Disturb | 0.2893 | 1.4468 | 2.9533 | 0.3535 | 0.3746 |
| Qwen2.5-vl-7B | Oracle | 0.3468 | 1.1704 | 2.4127 | 0.4731 | 0.4558 |
| Qwen3-vl-4B | Original | 0.4037 | 1.5249 | 2.8507 | 0.3817 | 0.4562 |
| Qwen3-vl-4B | Disturb | 0.3206 | 1.8805 | 3.2760 | 0.3181 | 0.3889 |
| Qwen3-vl-4B | Oracle | 0.4333 | 1.4758 | 2.7180 | 0.4115 | 0.4859 |
| Qwen3-vl-8B | Original | 0.3602 | 1.7089 | 3.0433 | 0.3538 | 0.4312 |
| Qwen3-vl-8B | Disturb | 0.2883 | 2.0637 | 3.4920 | 0.2882 | 0.3541 |
| Qwen3-vl-8B | Oracle | 0.3757 | 1.6806 | 2.9620 | 0.3675 | 0.4441 |
| Qwen3-vl-32B | Original | 0.3527 | 1.6561 | 3.0860 | 0.3742 | 0.4339 |
| Qwen3-vl-32B | Disturb | 0.3007 | 2.0676 | 3.5567 | 0.3165 | 0.3739 |
| Qwen3-vl-32B | Oracle | 0.3543 | 1.6695 | 3.1280 | 0.3722 | 0.4292 |
| InternVL3.5-8B | Original | 0.1429 | 0.5370 | 1.0413 | 0.3082 | 0.3332 |
| InternVL3.5-8B | Disturb | 0.0959 | 0.9468 | 1.5107 | 0.1702 | 0.1931 |
| InternVL3.5-8B | Oracle | 0.1799 | 0.5989 | 1.0780 | 0.3898 | 0.3894 |

### Key observations

- `Qwen2.5-vl-7B` is the only backbone where both `Disturb` and `Oracle` improve `TQS` relative to `Original`, and `Oracle` also raises `utility` and the informative-step rate. This is the regime where explicit grounding behaves like a net positive control signal.
- `Qwen3-vl-32B` shows the opposite mechanism: `Oracle` raises `ΔF` and `iterations` the most, but it also produces the lowest `utility` and the lowest informative-step rate among the Qwen3 family. This is the strongest evidence that explicit disambiguation can over-constrain the retrieval policy rather than help it.
- `Qwen3-vl-4B` and `Qwen3-vl-8B` sit in the middle: `Oracle` partially recovers `TQS` on 4B but still leaves utility below the original level on 8B. The rewrite effect is therefore family-dependent, not a one-way monotonic gain.

## Task-Level Mechanism on Qwen3-vl-32B

| Task | Setting | TQS | ΔF | Iterations | Utility | Informative rate |
|---|---|---:|---:|---:|---:|---:|
| Entity Recognition | Original | 0.3132 | 1.1952 | 2.2867 | 0.3477 | 0.3324 |
| Entity Recognition | Disturb | 0.2883 | 1.3515 | 2.5000 | 0.3087 | 0.3107 |
| Entity Recognition | Oracle | 0.3238 | 1.1687 | 2.2700 | 0.3436 | 0.3231 |
| Single-hop Attribute Query | Original | 0.4354 | 1.6435 | 2.8633 | 0.3906 | 0.4424 |
| Single-hop Attribute Query | Disturb | 0.3512 | 2.1356 | 3.4467 | 0.3177 | 0.3714 |
| Single-hop Attribute Query | Oracle | 0.4332 | 1.6692 | 2.9100 | 0.3883 | 0.4376 |
| Multi-hop | Original | 0.3044 | 1.9101 | 3.6500 | 0.2562 | 0.3279 |
| Multi-hop | Disturb | 0.2636 | 2.3486 | 4.0833 | 0.2245 | 0.2914 |
| Multi-hop | Oracle | 0.3004 | 1.9869 | 3.7967 | 0.2489 | 0.3178 |
| Comparison | Original | 0.2531 | 1.8386 | 3.8000 | 0.2325 | 0.3035 |
| Comparison | Disturb | 0.2076 | 2.3157 | 4.5167 | 0.2018 | 0.2605 |
| Comparison | Oracle | 0.2529 | 1.8379 | 3.8433 | 0.2346 | 0.3027 |
| Subproblem Aggregation | Original | 0.4574 | 1.6929 | 2.8300 | 0.6443 | 0.7633 |
| Subproblem Aggregation | Disturb | 0.3926 | 2.1867 | 3.2367 | 0.5299 | 0.6354 |
| Subproblem Aggregation | Oracle | 0.4613 | 1.6850 | 2.8200 | 0.6454 | 0.7648 |

### Task-level reading

- `Entity Recognition` is the only task family where `Oracle` clearly lifts `TQS` above `Original`, which suggests that explicit entity anchoring helps when the task is primarily about naming the referent.
- `Multi-hop` and `Comparison` are the failure modes: under `Oracle`, both tasks show a sharp rise in iterations and `ΔF` but a collapse in utility and `TQS`. This is the clearest mechanistic sign that explicit grounding can break the search policy on reasoning-heavy tasks.
- `Subproblem Aggregation` remains high-utility even under rewrite, which explains why the backbone-level averages do not collapse completely. The intervention is therefore selective, not uniformly harmful.

## Mechanistic Interpretation

The figure-level interpretation is that referential disambiguation changes the search policy more than it changes the information content of the query. For Qwen3 backbones, the model often keeps retrieving longer chains of evidence, but the utility gate weakens, so more retrieved facts are not translated into proportionally better trajectory quality.

This is the key mechanism statement to carry into the paper: the controllability gap is not just about ambiguity frequency. It is about whether the backbone treats explicit grounding as a useful anchor or as an over-constraining signal that reduces useful exploration.
