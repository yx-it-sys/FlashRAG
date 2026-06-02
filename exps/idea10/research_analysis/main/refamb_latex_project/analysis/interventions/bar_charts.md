# Bar Chart Analysis

# Cross-Backbone Bar Chart Analysis

This report analyzes the grouped bar charts for two metrics:

* `RTE`: retrieval-involved trajectory efficiency, measuring trajectory activity and process quality.
* `LJ Score`: final answer quality / correctness judged by the evaluation model.

The plotted settings are:

* `Original`: the base ReAct-style trajectory.
* `Disturb`: the first-round disturbance rewrite, where originally unambiguous queries are made more referentially ambiguous.
* `Oracle`: the oracle rewrite, where ambiguous queries are replaced with canonical entity-grounded expressions.

The compared backbones are:

* `Qwen2.5-VL-7B`
* `Qwen3-VL-4B`
* `Qwen3-VL-8B`
* `Qwen3-VL-32B`
* `InternVL3.5-8B`

## Main Phenomena

### 1. Intervention effects are strongly backbone-dependent

The intervention effects are not monotonic across backbones. The same query-side intervention can produce different changes depending on the model family, scale, and retrieval policy.

For RTE, `Disturb` generally suppresses trajectory activity across the evaluated backbones. This is visible not only in the Qwen3-VL family but also in Qwen2.5-VL-7B and InternVL3.5-8B. Therefore, the aggregate evidence does not support the claim that ambiguity injection universally improves trajectory activity. Instead, Disturb should be interpreted as a perturbative intervention that can stimulate exploration in selected cases or subsets, but whose overall effect on RTE is often negative.

`Oracle` shows a different pattern. On Qwen2.5-VL-7B, Oracle improves RTE relative to Original, indicating that explicit entity grounding can help a weaker backbone maintain more efficient retrieval trajectories. However, as the backbone becomes stronger within the Qwen3-VL family, the RTE gain brought by Oracle becomes weaker. On Qwen3-VL-32B, Oracle no longer produces a large RTE improvement and is close to the Original trajectory activity level. This suggests that stronger models may rely less on explicit entity grounding to maintain retrieval activity, or may use retrieved evidence more selectively.

Overall, query explicitness is not a backbone-agnostic control knob. Making a query more ambiguous or more explicit does not produce a universally predictable effect; its impact depends on the backbone's internal reasoning ability, retrieval policy, and evidence-utilization behavior.

### 2. Disturb consistently damages RTE more than it damages LJ

A clear trend across the bar charts is that Disturb often reduces RTE, while the corresponding degradation in LJ is weaker. This indicates that trajectory activity and final answer quality are not affected in the same way.

This is especially important for interpreting Disturb. Ambiguity injection can reduce the efficiency or usefulness of intermediate retrieval steps, but the final answer may remain relatively stable if the model can compensate through internal knowledge, visual grounding, or later evidence aggregation. Therefore, Disturb should not be understood simply as causing answer failure. Its stronger and more direct effect is on trajectory dynamics.

### 3. Oracle's RTE benefit decreases as model capability increases

Within the Qwen series, Oracle provides its clearest RTE benefit on Qwen2.5-VL-7B. For Qwen3-VL-4B and Qwen3-VL-8B, Oracle still often remains competitive with or slightly above Original, but the improvement becomes less salient. For Qwen3-VL-32B, the gap between Oracle and Original becomes small.

This suggests a capability-dependent trend: weaker models benefit more from canonical entity grounding because they need explicit retrieval-side anchors to sustain the trajectory. Stronger models, however, may already maintain sufficient entity grounding or answer-level reasoning under the Original setting, so Oracle contributes less additional trajectory activity.

This finding refines the interpretation of Oracle. Oracle is not merely a universal upper bound for trajectory activity. It is more accurately an intervention that reveals how much a model depends on explicit entity grounding. As model capability increases, this dependency appears to weaken.

### 4. LJ degradation from interventions becomes weaker on stronger models

For final answer quality, the negative effects of Disturb and Oracle are less severe on stronger backbones. In weaker or less stable models, perturbing the first-round query can more easily propagate into final answer errors. In stronger models, even when RTE decreases, LJ can remain stable or even slightly improve.

This indicates that stronger backbones may be better at decoupling final answer quality from noisy or suboptimal retrieval dynamics. They can tolerate lower trajectory activity because they possess stronger internal reasoning, better visual grounding, or more robust evidence utilization.

Thus, increasing model capability appears to reduce the dependence of final answer correctness on high trajectory activity.

### 5. RTE and LJ are partially decoupled

RTE and LJ do not always move together. A higher RTE does not necessarily imply a better final answer, and a lower RTE does not necessarily imply answer failure.

For example, a model may retrieve more novel evidence and maintain a more active trajectory, but fail to integrate that evidence into the final answer. Conversely, a stronger model may exhibit lower RTE after intervention while still preserving or improving LJ, because it can solve the task with fewer useful retrieval steps or stronger internal knowledge.

This supports the interpretation of RTE as a trajectory-dynamics diagnostic rather than a direct proxy for answer correctness. RTE measures whether the pipeline remains active and information-seeking, while LJ measures whether the final response is correct. The two are related but not equivalent.

### 6. Stronger models appear less dependent on exploratory trajectory activity

A broader pattern emerges across the Qwen series: as the model becomes stronger, final answer quality becomes less tightly coupled to exploratory trajectory activity. Qwen3-VL-32B does not require the largest RTE to achieve strong LJ performance. This suggests that stronger models may rely less on broad retrieval exploration and more on internal reasoning, stronger visual grounding, or selective evidence use.

This is a nontrivial finding for agentic RAG. It implies that increasing retrieval activity is not always the right objective, especially for stronger backbones. For weaker models, additional retrieval exploration may compensate for insufficient internal reasoning. For stronger models, however, excessive or perturbed retrieval may reduce trajectory efficiency without providing proportional answer-level benefits.

## Revised Findings for the Paper

### Finding 1: Query-side interventions are not backbone-agnostic

Disturb and Oracle do not produce uniform effects across models. Disturb generally suppresses RTE, while Oracle improves RTE most clearly on weaker backbones and becomes less beneficial as model capability increases. Therefore, query ambiguity and query explicitness should be treated as model-dependent control variables rather than universally beneficial or harmful factors.

### Finding 2: Trajectory activity and answer quality are only partially coupled

The changes in RTE and LJ are not always synchronized. Interventions can reduce trajectory activity while leaving final answer quality stable, or increase trajectory activity without producing proportional answer-quality gains. This confirms that RTE captures process-level dynamics rather than merely approximating final correctness.

### Finding 3: Stronger backbones rely less on exploratory retrieval

As model capability increases, the need for high trajectory activity appears to decrease. Stronger models can often preserve final answer quality even when interventions reduce RTE. This suggests that exploration-driven retrieval activity is more useful for weaker models and less reliably beneficial for stronger ones.

### Finding 4: Oracle measures entity-grounding dependency rather than a universal upper bound

Oracle is often treated as an upper-bound intervention because it supplies canonical entity information. However, the cross-backbone results show that Oracle does not universally maximize RTE. Its benefit is largest when the backbone needs explicit entity grounding to sustain retrieval, and weaker when the model already has strong grounding or reasoning capability.

## Concise Takeaway

Query-side clarification and ambiguity injection are not universal improvement strategies. Disturb generally reduces trajectory activity, while Oracle improves RTE mainly when the backbone depends on explicit entity grounding. As model capability increases, final answer quality becomes less tightly coupled to exploratory trajectory activity, revealing a partial decoupling between process efficiency and answer correctness.

## Suggested Paper Paragraph

Cross-backbone results further show that query-side interventions are not backbone-agnostic. As shown in Figure~\ref{fig:cross_backbone_intervention}, Disturb generally suppresses RTE across the evaluated backbones, indicating that ambiguity injection often weakens trajectory efficiency at the aggregate level, even though it may still recover selected failed trajectories. Oracle exhibits a more capability-dependent pattern: it improves RTE most clearly on Qwen2.5-VL-7B, but its benefit becomes less pronounced across stronger Qwen3-VL models. This suggests that precise entity grounding is most useful when the backbone relies on explicit retrieval-side anchors, whereas stronger models may already possess sufficient grounding or reasoning ability under the Original setting.

Importantly, RTE and LJ do not always move synchronously. Interventions that reduce trajectory activity do not necessarily lead to proportional answer-quality degradation, especially for stronger backbones. This indicates that RTE should be interpreted as a process-level diagnostic rather than a direct proxy for final correctness. Overall, the cross-backbone results suggest that referential ambiguity and entity explicitness function as model-dependent trajectory control variables: they reshape retrieval dynamics, but their conversion into final answer quality depends on the backbone's internal reasoning capacity and evidence-utilization behavior.

## Mechanism Supplement

The mechanism figures and tables are saved under [mechanism](/home/you/FlashRAG/exps/idea10/idea_reports/new/refamb_latex_project/figs/interventions/mechanism).

- [mechanism_overview_scatter.png](/home/you/FlashRAG/exps/idea10/idea_reports/new/refamb_latex_project/figs/interventions/mechanism/mechanism_overview_scatter.png)
- [mechanism_qwen3_vl_32b_task_breakdown.png](/home/you/FlashRAG/exps/idea10/idea_reports/new/refamb_latex_project/figs/interventions/mechanism/mechanism_qwen3_vl_32b_task_breakdown.png)

The key mechanism is a utility collapse, not a simple novelty gain. On the task-balanced summary, `Qwen3-vl-32B` moves from `Original` to `Oracle` as follows: `ΔF` rises from `1.6561` to `2.4592`, iterations rise from `3.0860` to `4.4151`, but utility drops from `0.3742` to `0.2927` and informative-step rate drops from `0.4339` to `0.3512`.

This is why `Oracle` can reduce `TQS` even when it looks more explicit: it increases retrieval effort while weakening the fraction of steps that are actually useful.

Task-level decomposition makes the mechanism sharper. On `Qwen3-vl-32B`, `Oracle` helps `Entity Recognition` (`TQS` `0.3132` -> `0.4262`) and keeps `Subproblem Aggregation` high, but it sharply degrades `Multi-hop` (`0.3044` -> `0.1289`) and `Comparison` (`0.2531` -> `0.1592`). The intervention is therefore task-selective, not globally beneficial or harmful.

The intervention effect is also not just a length effect. Because `Oracle` and `Disturb` do not change the VLM's historical query context, the relevant input to the retriever is altered without changing the agent's own policy trace. For `Qwen3-vl-32B`, the retrieval-side query form stays in a narrow band (`6.5877` for `Original`, `6.5293` for `Oracle`) while the utility and iteration profile changes substantially. This supports the claim that the main intervention channel is controllability, not prompt length.
