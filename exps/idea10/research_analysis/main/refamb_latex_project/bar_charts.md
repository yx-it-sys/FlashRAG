# Trajectory-Quality Bar Charts

This note summarizes the grouped bar charts under [`figs/interventions/bar_charts`](./figs/interventions/bar_charts) and is written in a paper-ready style.

## What the figure shows

The figure compares three settings across five multimodal backbones:

- `Original`: the unmodified retrieval trajectory.
- `Disturb`: the first-round disturbance rewrite.
- `Oracle`: the oracle rewrite.

Two metrics are shown:

- `RTE` (`trajectory_quality_score`), which captures trajectory-level retrieval efficiency / quality.
- `LJ Score`, the answer-quality score from the downstream evaluation signal (`output.metric_score`).

The chart uses the same original benchmark universe for all three settings. For `Disturb` and `Oracle`, samples that are not rewritten keep their `Original` score, so the bars represent a full-set mean under a fixed denominator rather than a subset-only average. This is the right comparison for the main paper because it isolates the net intervention effect on the benchmark as a whole.

The plotted models are ordered as:

1. `Qwen2.5-vl-7B`
2. `Qwen3-vl-4B`
3. `Qwen3-vl-8B`
4. `Qwen3-vl-32B`
5. `InternVL3.5-8B`

## Global means

The figure is consistent with the following overall means:

| Model | Original RTE | Disturb RTE | Oracle RTE | Original LJ | Disturb LJ | Oracle LJ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-vl-7B | 0.3077 | 0.2893 | 0.3468 | 0.2847 | 0.2647 | 0.3287 |
| Qwen3-vl-4B | 0.4037 | 0.3206 | 0.4333 | 0.3887 | 0.3460 | 0.4287 |
| Qwen3-vl-8B | 0.3602 | 0.2883 | 0.3757 | 0.3707 | 0.3400 | 0.4073 |
| Qwen3-vl-32B | 0.3527 | 0.3007 | 0.3543 | 0.4067 | 0.3973 | 0.4207 |
| InternVL3.5-8B | 0.1429 | 0.0959 | 0.1799 | 0.1287 | 0.1087 | 0.1507 |

## Main empirical pattern

### 1. The effect of clarification is backbone-dependent, not monotonic

The same retrieval-side rewrite does not induce the same response across models.

- On `Qwen2.5-vl-7B`, `Disturb` slightly lowers `RTE`, while `Oracle` raises both `RTE` and `LJ Score`.
- On the `Qwen3` family, `Disturb` consistently suppresses `RTE`, whereas `Oracle` often recovers part of that loss and, in several cases, improves answer quality as well.
- `InternVL3.5-8B` sits in the low-activity regime: both interventions change the trajectory more noticeably relative to its much smaller baseline, and `Oracle` produces the largest lift.

The important point is that retrieval-side clarification is not a universal “make the model better” knob. Its effect depends on the retrieval policy already encoded in the backbone.

### 2. Trajectory efficiency and answer quality are only partially coupled

`RTE` and `LJ Score` do not move in lockstep.

- `Qwen3-vl-32B` is the clearest example: `Oracle` keeps `RTE` roughly unchanged relative to `Original` while improving `LJ Score`.
- `Qwen3-vl-8B` and `Qwen3-vl-4B` also show regimes where `Oracle` improves both metrics, but the gain in answer quality is not proportional to the shift in trajectory efficiency.
- `Disturb` can reduce `RTE` without causing a commensurate collapse in `LJ Score`, which indicates that a simpler or shorter trajectory is not automatically a worse one.

This makes `RTE` useful as a process diagnostic, but not as a direct surrogate for final answer quality.

### 3. The strongest backbones are not uniformly the most stable

The higher-capacity `Qwen3` backbones do not simply dominate the weaker ones in intervention robustness.

- `Qwen3-vl-32B` is the most stable in terms of final quality, but it still exhibits a measurable trajectory reshaping under `Disturb` and `Oracle`.
- `Qwen3-vl-4B` is particularly sensitive in `RTE`, where `Disturb` causes a relatively large drop.

The model family therefore matters as much as model size. Capacity alone does not guarantee invariance to retrieval-side rewriting.

### 4. Task structure dominates the within-model profile

Across almost all backbones, the task families separate cleanly:

- `Subproblem Aggregation` consistently produces the highest `RTE` and the strongest `LJ Score`.
- `Comparison` and `Multi-hop` are more brittle, especially under `Disturb`, and show larger metric decoupling.
- `Entity Recognition` is the task most likely to benefit from `Oracle`, which is consistent with the idea that explicit referential grounding is most useful when the target entity itself is the main uncertainty.

This task-level pattern is important because it suggests the intervention is not globally beneficial or harmful. It is selectively useful depending on the reasoning structure of the question.

## Paper-ready interpretation

A concise interpretation for the main text is:

> Retrieval-side clarification is not a backbone-agnostic improvement strategy. Under a fixed benchmark universe, `Disturb` and `Oracle` reshape the retrieval trajectory differently across models: Qwen2.5 is relatively receptive to clarification, while Qwen3 backbones tend to suppress retrieval activity under disturbance and recover selectively under oracle rewriting. The resulting changes in trajectory efficiency are only partially aligned with answer quality, indicating that referential ambiguity primarily acts as a controllability variable rather than a simple accuracy bottleneck.

## Caption-style summary

> Grouped bar charts of trajectory efficiency (`RTE`) and answer quality (`LJ Score`) across five multimodal backbones. All settings are averaged over the same benchmark universe; for `Disturb` and `Oracle`, non-rewritten samples retain their `Original` score, so each bar reflects the net effect of the intervention under a fixed denominator. The figure shows that clarification is not monotonic: it is mildly beneficial on some backbones and tasks, but suppressive or only weakly beneficial on others, with the largest gains concentrated in entity-centric questions.

## Recommended citation-style phrasing

If you need a short sentence in the paper:

> Across backbones, retrieval-side rewriting has a non-monotonic effect on trajectory quality: it can increase retrieval activity on some models, suppress it on others, and its impact on final answer quality is only partially correlated with process efficiency.

