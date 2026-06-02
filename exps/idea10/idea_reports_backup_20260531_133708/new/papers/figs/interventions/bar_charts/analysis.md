# Bar Chart Analysis

This directory contains grouped bar charts for two metrics:

- `RTE` (`trajectory_quality_score`): retrieval-involved trajectory efficiency.
- `LJ Score`: final answer quality / correctness from the underlying metric score.

The plotted settings are:

- `Original`: the base trajectory.
- `Disturb`: the first-round disturbance rewrite.
- `Oracle`: the oracle rewrite.

The model sets used in the figures are:

- `InternVL3.5-8B` vs `Qwen3-vl-4B`
- `Qwen2.5-vl-7B`, `Qwen3-vl-4B`, `Qwen3-vl-8B`
- `Qwen2.5-vl-7B`, `Qwen3-vl-4B`, `Qwen3-vl-8B`, `Qwen3-vl-32B`
- `InternVL3.5-8B`, `Qwen2.5-vl-7B`, `Qwen3-vl-4B`, `Qwen3-vl-8B`, `Qwen3-vl-32B`

`Qwen3-vl-2B` is not included here because the corresponding ambiguity-labeled trajectories are missing in this checkout.

## Overall Means

| Model | Original RTE | Disturb RTE | Oracle RTE | Original LJ | Disturb LJ | Oracle LJ |
|---|---:|---:|---:|---:|---:|---:|
| InternVL3.5-8B | 0.1429 | 0.1302 | 0.1567 | 0.1287 | 0.1116 | 0.1081 |
| Qwen2.5-vl-7B | 0.3077 | 0.3808 | 0.3370 | 0.2847 | 0.2905 | 0.2261 |
| Qwen3-vl-4B | 0.4037 | 0.3503 | 0.3752 | 0.3887 | 0.3508 | 0.3525 |
| Qwen3-vl-8B | 0.3602 | 0.3210 | 0.3199 | 0.3707 | 0.3594 | 0.3193 |
| Qwen3-vl-32B | 0.3527 | 0.3136 | 0.2758 | 0.4067 | 0.3900 | 0.4175 |

## Main Phenomena

### 1. Clarification is not monotonic

The same retrieval-side clarification can increase, decrease, or barely change trajectory efficiency depending on backbone family.

- `Qwen2.5-vl-7B` is the only model where `Disturb` clearly increases mean `RTE` over `Original` (`+23.7%`), suggesting that explicit retrieval-side grounding can stimulate more retrieval activity on this backbone.
- All `Qwen3` models move in the opposite direction: both `Disturb` and `Oracle` generally suppress `RTE`, and the suppression becomes stronger as the backbone gets larger.
- `Qwen3-vl-32B` shows the strongest `Oracle` suppression of `RTE` among the Qwen3 family, even though it is the strongest backbone in the set.

This is the first nontrivial point: retrieval-side explicitness is not a universal gain. The intervention effect depends on the retrieval policy learned by the backbone.

### 2. Process quality and answer quality are decoupled

`RTE` and `LJ Score` do not move together consistently.

- `Qwen3-vl-32B`: `Oracle` lowers `RTE` from `0.3527` to `0.2758`, but `LJ Score` rises from `0.4067` to `0.4175`.
- `Qwen2.5-vl-7B`: `Disturb` raises `RTE`, but `LJ Score` barely changes. More retrieval activity does not necessarily translate into better answers.
- `Qwen3-vl-8B`: both `Disturb` and `Oracle` reduce `RTE` and also reduce `LJ Score`, showing a coupled degradation regime.

So `RTE` should be interpreted as a trajectory-dynamics diagnostic, not as a proxy for final correctness.

### 3. The intervention response flips across backbone generations

There is a clear family-level shift:

- `Qwen2.5-vl-7B` reacts to explicit retrieval-side grounding by becoming more active.
- `Qwen3-vl-4B`, `Qwen3-vl-8B`, and `Qwen3-vl-32B` generally react by becoming less active.
- The direction of the `Oracle` effect changes from positive/neutral to negative as the backbone moves from Qwen2.5 to Qwen3.

This is a useful paper-level observation because it implies that the same intervention cannot be deployed as a backbone-agnostic recipe.

### 4. Task sensitivity is highly non-uniform

Across all models and settings, `Subproblem Aggregation` is the highest-activity task family:

- It consistently has the largest `RTE`.
- It also tends to have the strongest `LJ Score`.

This means compositional, multi-step queries are both the most retrieval-intensive and the most exploitable by the models. They are not just harder; they are structurally more “alive” in the trajectory.

By contrast:

- `Comparison` often has middling `RTE` but weak `LJ Score`, especially for the Qwen3 family.
- `Multi-hop` is often the most fragile under `Disturb` and `Oracle` on Qwen3 backbones.

### 5. A task-specific counterexample on the strongest backbone

`Qwen3-vl-32B` is the most interesting case:

- `Oracle` drops `RTE` by `21.8%` overall relative to `Original`.
- Yet `LJ Score` increases by `2.7%`.
- On `Entity Recognition`, `Oracle` improves both `RTE` and `LJ Score` strongly.
- On `Comparison`, `Oracle` sharply lowers `RTE` but raises `LJ Score` substantially.

This is a stronger claim than “rewrite helps” or “rewrite hurts”: the retrieval-side intervention changes the *shape* of the trajectory, but not in a way that is uniformly aligned with final answer quality.

## Novelty-Driven Findings

The main novelty is not simply that ambiguity matters, but that **ambiguity reduction and retrieval controllability are partially orthogonal**.

### Novel finding 1: Low ambiguity does not imply high controllability

From the ambiguity statistics computed separately:

- `Qwen3-vl-32B` has the lowest entity-ambiguity frequency.
- Yet it shows the largest `Oracle`-induced suppression of `RTE`.

This is important because it breaks the intuitive assumption that a less ambiguous retrieval input should be easier to control. In practice, the backbone can still become brittle to explicit retrieval-side clarification even when ambiguity is already rare.

### Novel finding 2: “More explicit” can mean “less exploratory”

On `Qwen3` backbones, `Oracle` often reduces trajectory activity.

That suggests the retrieval-side clarification may over-constrain the retrieval policy:

- the model stops exploring alternate evidence paths,
- but the final answer can stay stable or even improve slightly.

This is a more interesting regime than standard “rewrite improves retrieval” narratives, because it reveals a controllability gap at the query--retriever interface.

### Novel finding 3: The same intervention can be activity-positive on one backbone and activity-negative on another

`Disturb` increases `RTE` on `Qwen2.5-vl-7B`, but decreases `RTE` on all `Qwen3` models.

That sign flip is a strong backbone-dependent effect, and it is exactly the kind of result that supports the paper’s argument that query specificity is not an invariant control knob.

## Suggested Interpretation for the Paper

The bar charts support the following narrative:

1. `RTE` is sensitive to how retrieval-side clarification interacts with backbone policy.
2. The direction of that interaction changes across model generations.
3. Final answer quality only partly tracks trajectory quality.
4. Therefore, referential ambiguity is not just an error source to eliminate; it is a control variable whose usefulness depends on the model.

## Concise Takeaway

If this needs to be summarized in one sentence for the paper:

> Retrieval-side clarification is not a universally monotonic improvement strategy: on Qwen2.5 it can stimulate retrieval, while on Qwen3 it often suppresses retrieval activity, and the strongest backbone can even become less active yet slightly more accurate after oracle clarification.

## Mechanism Supplement

The mechanism figures and tables are saved under [mechanism](/home/you/FlashRAG/exps/idea10/idea_reports/new/papers/figs/interventions/mechanism).

- [mechanism_overview_scatter.png](/home/you/FlashRAG/exps/idea10/idea_reports/new/papers/figs/interventions/mechanism/mechanism_overview_scatter.png)
- [mechanism_qwen3_vl_32b_task_breakdown.png](/home/you/FlashRAG/exps/idea10/idea_reports/new/papers/figs/interventions/mechanism/mechanism_qwen3_vl_32b_task_breakdown.png)

The key mechanism is a utility collapse, not a simple novelty gain. On the task-balanced summary, `Qwen3-vl-32B` moves from `Original` to `Oracle` as follows: `ΔF` rises from `1.6561` to `2.4592`, iterations rise from `3.0860` to `4.4151`, but utility drops from `0.3742` to `0.2927` and informative-step rate drops from `0.4339` to `0.3512`.

This is why `Oracle` can reduce `TQS` even when it looks more explicit: it increases retrieval effort while weakening the fraction of steps that are actually useful.

Task-level decomposition makes the mechanism sharper. On `Qwen3-vl-32B`, `Oracle` helps `Entity Recognition` (`TQS` `0.3132` -> `0.4262`) and keeps `Subproblem Aggregation` high, but it sharply degrades `Multi-hop` (`0.3044` -> `0.1289`) and `Comparison` (`0.2531` -> `0.1592`). The intervention is therefore task-selective, not globally beneficial or harmful.

The intervention effect is also not just a length effect. Because `Oracle` and `Disturb` do not change the VLM's historical query context, the relevant input to the retriever is altered without changing the agent's own policy trace. For `Qwen3-vl-32B`, the retrieval-side query form stays in a narrow band (`6.5877` for `Original`, `6.5293` for `Oracle`) while the utility and iteration profile changes substantially. This supports the claim that the main intervention channel is controllability, not prompt length.
