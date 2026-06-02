# Mechanism Analysis for Pipeline Inactivation

This supplement supports the subsection *Inhibition Effect of Two Types of Interventions on Pipeline Inactivation* in `main.tex`.
The goal is not to restate the whole paper, but to explain *why* `Disturb` and `Oracle` interventions affect pipeline inactivation differently across backbones and task regimes.
The two tables below are recomputed with the same full-universe fallback convention used in the bar charts: for `Disturb` and `Oracle`, samples not rewritten by the intervention retain their `Original` values before task-balanced aggregation.

The analysis follows the same conceptual angle as the main text:

- pipeline inactivation is reflected by repeated queries, zero-information steps, low step utility, and late-iteration collapse of marginal gain;
- the interventions act only on the retrieval-side evidence pathway, so the VLM policy trace itself is held fixed;
- the key question is not whether an intervention increases retrieval activity, but whether the additional activity is converted into useful reasoning progress.

## Core mechanism

The updated statistics show that `Disturb` and `Oracle` do not simply add or remove information.
They change the *shape* of the retrieval-feedback loop.

Across backbones, `Disturb` reliably increases `\Delta F` and the number of iterations, but this increase is not beneficial by itself.
In almost every backbone, `Disturb` lowers utility and RTE, indicating that ambiguity-induced rewriting tends to transform stagnation from a low-novelty failure into a low-utility exploration failure.
The trajectory becomes longer and more fact-rich, but the additional evidence is not converted into proportionally useful reasoning steps.

`Oracle` behaves differently.
It improves RTE and utility when the backbone benefits from explicit entity grounding, as seen most clearly on `Qwen2.5-vl-7B` and `InternVL3.5-8B`.
However, as the backbone becomes stronger, the marginal effect of `Oracle` becomes smaller.
On `Qwen3-vl-32B`, `Oracle` leaves RTE, `\Delta F`, utility, and informative-step rate almost unchanged, suggesting a saturation regime rather than a strong recovery or a clear over-constraint effect.

The central mechanism is therefore the balance between *novelty* and *utility*.
An intervention is useful only when it increases or preserves factual novelty while maintaining the model's ability to exploit the retrieved evidence.
When utility weakens, additional retrieval activity becomes another form of inactivation: the loop continues to move, but the movement is not productive.

## Backbone-level pattern

| Model | Setting | RTE | Delta F | Iterations | Utility | Informative rate |
|---|---|---:|---:|---:|---:|---:|
| InternVL3.5-8B | Original | 0.1429 | 0.5369 | 1.0413 | 0.3082 | 0.3332 |
| InternVL3.5-8B | Disturb | 0.1031 | 1.2794 | 1.9681 | 0.1736 | 0.1949 |
| InternVL3.5-8B | Oracle | 0.2140 | 0.7904 | 1.3762 | 0.4075 | 0.3967 |
| Qwen2.5-vl-7B | Original | 0.3077 | 1.1593 | 2.6993 | 0.4354 | 0.4019 |
| Qwen2.5-vl-7B | Disturb | 0.2893 | 1.7330 | 3.1150 | 0.3977 | 0.4457 |
| Qwen2.5-vl-7B | Oracle | 0.3468 | 1.4234 | 2.4734 | 0.4712 | 0.5185 |
| Qwen3-vl-4B | Original | 0.4037 | 1.5249 | 2.8507 | 0.3817 | 0.4562 |
| Qwen3-vl-4B | Disturb | 0.3206 | 1.9907 | 3.4056 | 0.3338 | 0.4126 |
| Qwen3-vl-4B | Oracle | 0.4333 | 1.9025 | 3.2453 | 0.3896 | 0.4805 |
| Qwen3-vl-8B | Original | 0.3602 | 1.7089 | 3.0433 | 0.3538 | 0.4312 |
| Qwen3-vl-8B | Disturb | 0.2883 | 2.3103 | 3.8140 | 0.3012 | 0.3755 |
| Qwen3-vl-8B | Oracle | 0.3757 | 2.1898 | 3.6092 | 0.2981 | 0.3867 |
| Qwen3-vl-32B | Original | 0.3527 | 1.6561 | 3.0860 | 0.3742 | 0.4339 |
| Qwen3-vl-32B | Disturb | 0.3007 | 2.0676 | 3.5567 | 0.3165 | 0.3739 |
| Qwen3-vl-32B | Oracle | 0.3543 | 1.6695 | 3.1280 | 0.3722 | 0.4292 |

### Compact deltas relative to `Original` (show in paper)

| Model | $\Delta$Disturb RTE | $\Delta$Oracle RTE | $\Delta$Disturb $\Delta F$ | $\Delta$Oracle $\Delta F$ | $\Delta$Disturb Utility | $\Delta$Oracle Utility |
|---|---:|---:|---:|---:|---:|---:|
| InternVL3.5-8B | -0.0398 | +0.0711 | +0.7424 | +0.2535 | -0.1346 | +0.0993 |
| Qwen2.5-vl-7B | -0.0184 | +0.0391 | +0.5737 | +0.2641 | -0.0377 | +0.0358 |
| Qwen3-vl-4B | -0.0831 | +0.0296 | +0.4658 | +0.3776 | -0.0479 | +0.0079 |
| Qwen3-vl-8B | -0.0719 | +0.0155 | +0.6014 | +0.4809 | -0.0526 | -0.0557 |
| Qwen3-vl-32B | -0.0520 | +0.0016 | +0.4115 | +0.0134 | -0.0577 | -0.0021 |

### Reading the table

- `Disturb` consistently increases factual novelty and the number of retrieval iterations, but it lowers RTE and utility across all evaluated backbones. This is the clearest signature of low-utility exploration: the trajectory becomes more active in a raw retrieval sense, yet less effective as a reasoning process.
- `Oracle` produces genuine recovery on `Qwen2.5-vl-7B` and `InternVL3.5-8B`. It improves RTE, utility, and informative-step rate, indicating that explicit entity grounding can inhibit pipeline inactivation when the backbone needs a stronger retrieval anchor.
- `Qwen3-vl-4B` and `Qwen3-vl-8B` occupy an intermediate regime. `Oracle` can improve RTE, but the improvement is not uniformly reflected in utility or informative-step rate. This suggests partial recovery rather than complete restoration of trajectory efficiency.
- `Qwen3-vl-32B` shows a saturation regime. `Oracle` changes the trajectory only marginally: RTE, `\Delta F`, iterations, utility, and informative-step rate all remain close to `Original`. This does not support a strong over-constraint interpretation; rather, it suggests that the strongest backbone already has sufficient grounding or evidence-selection ability under the Original setting, leaving little room for oracle clarification to improve trajectory dynamics.
- `InternVL3.5-8B` should be interpreted as a low-trajectory-activity regime in this experiment, not as a low-capacity model. Its response shows that `Oracle` can restore trajectory quality when the original retrieval loop is weak, whereas `Disturb` destabilizes the same loop.

## Mechanistic interpretation

The important mechanism is not that `Disturb` or `Oracle` simply add more facts.
They rewire the retrieval-feedback loop by changing the relation between factual novelty and utility.

`Disturb` is the clearest case of an inactivation-mode shift.
It reduces the original form of inactivation associated with low novelty by increasing `\Delta F` and extending the retrieval loop.
However, it induces another form of inactivation: low-utility exploration.
The model retrieves more information, but the additional evidence is less consistently transformed into useful intermediate reasoning.
Thus, the loop remains active in a superficial sense while becoming less productive.

`Oracle` is more selective.
It inhibits inactivation when explicit entity grounding simultaneously improves factual novelty or preserves novelty while increasing utility, as seen in `Qwen2.5-vl-7B` and `InternVL3.5-8B`.
For stronger Qwen3 backbones, the same intervention yields smaller gains, and on `Qwen3-vl-32B` it nearly saturates.
This suggests that oracle clarification is most useful when the model still depends on external entity anchors; once the backbone already has sufficient grounding or evidence-selection ability, the marginal trajectory benefit of explicit entity rewriting becomes small.

This explains why the main paper should not describe intervention effects only as ambiguity removal or ambiguity injection.
The relevant question is whether an intervention preserves the balance between retrieval novelty and reasoning utility.
A successful intervention suppresses pipeline inactivation by interrupting repeated or low-information retrieval while keeping the retrieved evidence usable.
An unsuccessful intervention merely changes the form of inactivation, producing longer and more fact-rich trajectories that remain less useful for reasoning.

The updated statistics therefore sharpen the main conclusion:
referential ambiguity is a control variable at the query--retriever interface, but the effect of controlling it depends on the backbone's latent retrieval policy.
The interventions can suppress pipeline inactivation, but the success of that suppression is backbone- and task-dependent rather than universal.

