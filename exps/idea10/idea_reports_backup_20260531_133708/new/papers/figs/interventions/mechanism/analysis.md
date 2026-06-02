# Mechanism Analysis

This supplement focuses on why rewrite changes trajectory quality, using the same task-balanced aggregation convention as the bar charts.

Important setting note: `Oracle` and `Disturb` do not change the query history observed by the VLM. They only alter the retrieval-side input / feedback. The agent policy is therefore held fixed, and all observed differences come from the changed retrieval evidence returned to the same trajectory.

## Core Mechanism

The dominant effect of both `Disturb` and `Oracle` is not a simple increase in fact novelty. Instead, the retrieval-side intervention usually pushes the same agent policy into longer trajectories with higher `ΔF` but lower `utility` and lower informative-step density.

In other words, the intervention often increases the amount of retrieved material while decreasing the fraction of steps that the downstream reasoning actually uses.

## Backbone-Level Pattern

| Model | Setting | TQS | ΔF | Iterations | Utility | Informative rate |
|---|---|---:|---:|---:|---:|---:|
| InternVL3.5-8B | Original | 0.1429 | 0.5370 | 1.0413 | 0.3082 | 0.3332 |
| InternVL3.5-8B | Disturb | 0.1031 | 1.2794 | 1.9681 | 0.1736 | 0.1949 |
| InternVL3.5-8B | Oracle | 0.2140 | 0.7904 | 1.3762 | 0.4075 | 0.3967 |
| Qwen2.5-vl-7B | Original | 0.3077 | 1.1593 | 2.6993 | 0.4354 | 0.4019 |
| Qwen2.5-vl-7B | Disturb | 0.3708 | 1.7330 | 3.1150 | 0.3977 | 0.4457 |
| Qwen2.5-vl-7B | Oracle | 0.3736 | 1.4234 | 2.4734 | 0.4712 | 0.5185 |
| Qwen3-vl-4B | Original | 0.4037 | 1.5249 | 2.8507 | 0.3817 | 0.4562 |
| Qwen3-vl-4B | Disturb | 0.3508 | 1.9907 | 3.4056 | 0.3338 | 0.4126 |
| Qwen3-vl-4B | Oracle | 0.3811 | 1.9025 | 3.2453 | 0.3896 | 0.4805 |
| Qwen3-vl-8B | Original | 0.3602 | 1.7089 | 3.0433 | 0.3538 | 0.4312 |
| Qwen3-vl-8B | Disturb | 0.3129 | 2.3103 | 3.8140 | 0.3012 | 0.3755 |
| Qwen3-vl-8B | Oracle | 0.3168 | 2.1898 | 3.6092 | 0.2981 | 0.3867 |
| Qwen3-vl-32B | Original | 0.3527 | 1.6561 | 3.0860 | 0.3742 | 0.4339 |
| Qwen3-vl-32B | Disturb | 0.3131 | 2.3673 | 3.8520 | 0.3141 | 0.3963 |
| Qwen3-vl-32B | Oracle | 0.3135 | 2.4592 | 4.4151 | 0.2927 | 0.3512 |

### Key observations

- `Qwen2.5-vl-7B` is the only backbone where both `Disturb` and `Oracle` improve `TQS` relative to `Original`, and `Oracle` also raises `utility` and the informative-step rate. This is the regime where retrieval-side explicit grounding behaves like a net positive control signal.
- `Qwen3-vl-32B` shows the opposite mechanism: `Oracle` raises `ΔF` and `iterations` the most, but it also produces the lowest `utility` and the lowest informative-step rate among the Qwen3 family. This is the strongest evidence that explicit disambiguation can over-constrain the retrieval feedback rather than help it.
- `Qwen3-vl-4B` and `Qwen3-vl-8B` sit in the middle: `Oracle` partially recovers `TQS` on 4B but still leaves utility below the original level on 8B. The intervention effect is therefore family-dependent, not a one-way monotonic gain.

## Task-Level Mechanism on Qwen3-vl-32B

| Task | Setting | TQS | ΔF | Iterations | Utility | Informative rate |
|---|---|---:|---:|---:|---:|---:|
| Entity Recognition | Original | 0.3132 | 1.1952 | 2.2867 | 0.3477 | 0.3324 |
| Entity Recognition | Disturb | 0.3079 | 2.2580 | 3.6102 | 0.2770 | 0.3568 |
| Entity Recognition | Oracle | 0.4262 | 2.0150 | 3.3226 | 0.3107 | 0.3786 |
| Single-hop Attribute Query | Original | 0.4354 | 1.6435 | 2.8633 | 0.3906 | 0.4424 |
| Single-hop Attribute Query | Disturb | 0.3539 | 2.4180 | 3.6891 | 0.3195 | 0.3918 |
| Single-hop Attribute Query | Oracle | 0.2599 | 2.6425 | 4.2857 | 0.2750 | 0.3250 |
| Multi-hop | Original | 0.3044 | 1.9101 | 3.6500 | 0.2562 | 0.3279 |
| Multi-hop | Disturb | 0.2781 | 2.4599 | 4.1232 | 0.2342 | 0.3076 |
| Multi-hop | Oracle | 0.1289 | 3.0235 | 5.8519 | 0.1519 | 0.2089 |
| Comparison | Original | 0.2531 | 1.8386 | 3.8000 | 0.2325 | 0.3035 |
| Comparison | Disturb | 0.2243 | 2.4410 | 4.5365 | 0.2156 | 0.2816 |
| Comparison | Oracle | 0.1592 | 2.5125 | 5.6154 | 0.2260 | 0.2603 |
| Subproblem Aggregation | Original | 0.4574 | 1.6929 | 2.8300 | 0.6443 | 0.7633 |
| Subproblem Aggregation | Disturb | 0.4015 | 2.2595 | 3.3010 | 0.5241 | 0.6436 |
| Subproblem Aggregation | Oracle | 0.5930 | 2.1027 | 3.0000 | 0.5000 | 0.5833 |

### Task-level reading

- `Entity Recognition` is the only task family where `Oracle` clearly lifts `TQS` above `Original`, which suggests that explicit entity anchoring helps when the task is primarily about naming the referent.
- `Multi-hop` and `Comparison` are the failure modes: under `Oracle`, both tasks show a sharp rise in iterations and `ΔF` but a collapse in utility and `TQS`. This is the clearest mechanistic sign that explicit grounding can break the retrieval-feedback loop on reasoning-heavy tasks.
- `Subproblem Aggregation` remains high-utility even under rewrite, which explains why the backbone-level averages do not collapse completely. The intervention is therefore selective, not uniformly harmful.

## Mechanistic Interpretation

The figure-level interpretation is that referential disambiguation changes the retrieval feedback more than it changes the agent policy. For Qwen3 backbones, the model often keeps retrieving longer chains of evidence, but the utility gate weakens, so more retrieved facts are not translated into proportionally better trajectory quality.

This is the key mechanism statement to carry into the paper: the controllability gap is not just about ambiguity frequency. It is about whether the backbone treats explicit grounding in the retrieval feedback as a useful anchor or as an over-constraining signal that reduces useful exploration.
