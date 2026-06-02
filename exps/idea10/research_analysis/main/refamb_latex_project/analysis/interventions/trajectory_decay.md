# Trajectory-Level Step-Score Decay

This note corresponds to the figure saved under [true_trajectory_decay](/home/you/FlashRAG/exps/idea10/research_analysis/main/refamb_latex_project/figs/interventions/true_trajectory_decay).

## Definition

The plotted curve is a true trajectory-level decay curve:

* For each trajectory, we take its per-step `step_score` sequence.
* Steps after termination are filled with `0`.
* The plotted value at step `t` is the average over all trajectories after zero-padding.

This is different from the earlier cross-sectional curve, which only averages over trajectories that are still alive at step `t`.

We intentionally use this zero-padded convention because the target claim is about the overall exploration activity of the full ReAct trajectory as iteration proceeds, not only the conditional behavior of trajectories that remain alive. Under this view, early termination is itself part of the decay signal: when a trajectory stops exploring, its later steps should contribute no further activity. Zero-padding therefore gives the most direct estimate of the global exploration level at each round and makes the decreasing-trend claim literal.

## Main Observations

### 0. Trajectory Length Distribution

This table reports the proportion of items whose original trajectory length equals `0` to `5` iterations in each model's `Original` setting. Here, `0` means the trajectory terminated before producing any valid iteration step.

| Model          |      0 |      1 |      2 |      3 |      4 |      5 |
| -------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| Qwen2.5-vl-7B  | 12.80% | 18.07% | 14.40% | 20.20% | 10.20% | 24.33% |
| Qwen3-vl-4B    |  3.53% | 29.00% | 16.73% | 11.60% |  7.87% | 31.27% |
| Qwen3-vl-8B    |  5.20% | 21.13% | 18.80% | 11.33% |  6.07% | 37.47% |
| Qwen3-vl-32B   |  1.53% | 17.60% | 25.33% | 14.60% |  8.13% | 32.80% |
| InternVL3.5-8B | 48.80% | 27.27% | 10.20% |  4.67% |  2.87% |  6.20% |

#### Notes

* All rows are computed from the full `Original` universe of 1,500 items per model.
* The percentages sum to 100% within each model.
* The distribution highlights how often trajectories stop early versus reaching later rounds.
* This distribution should be read as auxiliary context for the zero-padded curves rather than as a direct measurement of per-step score.

### 1. Decay is sharp for all models

All backbones show a strong decline from step 1 to step 5 once termination is counted explicitly. This confirms that trajectory activity is not sustained uniformly over the full horizon.

The decay is especially clear after the second or third iteration, where the mean step score rapidly approaches zero under all settings. This supports the claim that trajectory inactivation is not an isolated phenomenon of a particular intervention or model, but a general temporal pattern of the ReAct-style retrieval loop when the full trajectory horizon is considered.

### 2. Disturb reshapes the decay curve rather than uniformly boosting activity

`Disturb` should not be interpreted as a simple early-step boost. In several models, including `Qwen2.5-vl-7B` and the Qwen3-VL family, the first-step score under `Disturb` is lower than `Original` or `Oracle`. However, `Disturb` often preserves relatively higher scores in the following rounds, especially around steps 2--4.

This indicates a delayed-decay pattern. The intervention does not make the trajectory globally stronger from the beginning; instead, it changes the temporal shape of activity. The trajectory may remain active for longer after the initial step, but this activity should not be directly equated with useful reasoning. This observation is consistent with the mechanism analysis: `Disturb` can increase raw retrieval movement while reducing retrieval utility.

### 3. Oracle mostly follows or slightly stabilizes the Original trajectory

`Oracle` does not uniformly preserve late-stage activity. On `Qwen2.5-vl-7B` and `InternVL3.5-8B`, `Oracle` improves the first-step score relative to `Original`, suggesting that explicit entity grounding can strengthen the initial retrieval state for backbones that benefit from clearer entity anchors. However, the curves still decay rapidly, indicating that early stabilization does not remove late-stage inactivation.

On `Qwen3-vl-32B`, `Oracle` largely follows the `Original` decay curve rather than producing a clear late-stage survival advantage. This suggests a saturation regime: the stronger backbone may already possess sufficient entity grounding or evidence-selection ability under the `Original` setting, leaving limited room for oracle clarification to reshape the full trajectory. The late-stage curve is not dominated by `Oracle`; in this model, `Disturb` more visibly preserves mid-to-late step scores, although this preservation may reflect low-utility exploration rather than productive reasoning.

### 4. InternVL3.5-8B remains the lowest trajectory-activity regime

`InternVL3.5-8B` exhibits much lower absolute step scores than the Qwen family, and the zero-padded decay reaches near-zero by the final step for all settings. This aligns with its trajectory length distribution, where a large proportion of items terminate before or shortly after the first valid iteration.

This model should therefore be interpreted as a low-trajectory-activity regime in the present experiment, rather than simply as a low-capacity model. Its behavior provides a useful reference point for understanding how early termination and weak step-level activity jointly shape the zero-padded decay curve.

## Takeaway

The zero-padded trajectory-level curve makes the decay claim explicit: once terminal trajectories are counted as inactive beyond their stopping point, the average step score decays rapidly across the full horizon for all models and settings.

The figure also refines the interpretation of the two interventions. `Disturb` does not provide a stable global boost; it often lowers the first-step score while delaying decay in subsequent rounds, producing a longer but not necessarily more useful activity pattern. `Oracle` can stabilize early activity on models that benefit from explicit entity anchoring, but it does not remove late-stage collapse and becomes nearly identical to `Original` on stronger backbones such as `Qwen3-vl-32B`.

This figure therefore complements the mechanism analysis rather than replacing it. The decay curves show that interventions can reshape the temporal profile of trajectory activity, but they do not eliminate the structural exhaustion of step-level reasoning utility. In particular, sustained step-score survival should not be interpreted as successful recovery unless the additional activity is also converted into useful reasoning progress.
