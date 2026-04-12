# Experiment Report: First-Round Oracle Rewrite for Entity-Ambiguous Queries

## Core Question

This experiment tests a narrow causal intervention on the current OmniSearch agent loop:

- Only consider items whose first text retrieval query is labeled as entity-ambiguous.
- Replace that first query with an oracle rewrite that explicitly names the target entity.
- Do not intervene in any later round.
- Let the agent continue its own iterative reasoning and retrieval from the updated context.

The goal is to measure how much of the current failure comes from the first unresolved query itself, rather than from downstream reasoning alone.

## Experimental Setup

### Data Scope

- Source experiment: `crag_mm_2026_03_31_14_06_experiment`
- Rewrite label source: `trajectory_annotation.llm_labeled.rewrite.jsonl`
- Selection rule:
  - the item has at least one rewrite label
  - the first labeled retrieval step is `step_index=3`
  - the first step is labeled `entity_ambiguous == "Yes"`
  - the first step has a non-empty `rewrite_query`

This yields:

- Total validation items: `1199`
- Items with any entity ambiguity: `405`
- Items with first-step entity ambiguity and oracle rewrite: `369`

### Intervention

For each selected item:

1. Run the same initial multimodal prompt as the original `OmniSearchPipeline`.
2. Let the model generate the first response normally.
3. If the first response asks for `Text Retrieval`, replace the generated first query with the oracle rewrite from annotation.
4. Execute retrieval using the oracle query.
5. Feed retrieval results back to the agent using the same follow-up message logic as the original pipeline.
6. Do not rewrite or constrain any later query.

This is therefore a first-round-only intervention, not a full oracle rollout.

### Result Artifacts

Experiment outputs are stored in:

- `/home/you/FlashRAG/exps/idea10/scripts/exps/results`

Key files:

- `intermediate_data.json`
- `omnisearch_trajectories.jsonl`
- `paired_comparison.json`
- `paired_comparison.csv`
- `run_summary.json`

## Main Results

### Coverage

- Subset size: `369`
- Oracle rewrite successfully applied: `366`

The remaining 3 cases did not actually execute the intended first-round text-query replacement path at runtime.

### Aggregate Comparison on the 369-item Subset

| Metric | Baseline | Oracle-Rewrite | Delta |
| --- | ---: | ---: | ---: |
| GPTAcc | 0.0515 | 0.1165 | +0.0650 |
| F1 | 0.1617 | 0.2158 | +0.0540 |
| Recall | 0.3014 | 0.3684 | +0.0670 |
| Precision | 0.1384 | 0.1878 | +0.0495 |
| Acc | 0.0027 | 0.0081 | +0.0054 |
| EM | 0.0000 | 0.0027 | +0.0027 |

Takeaway:

- The strongest headline result is the `GPTAcc` gain from `5.15%` to `11.65%`.
- This is a relative gain of more than `2x` on the targeted ambiguity subset.
- Lexical overlap metrics also improve consistently, which suggests the gain is not only due to judge noise.

## Outcome Distribution

### Status Transition

The most important trajectory-level shifts are:

- `missing_final_answer -> ok`: `54`
- `generation_error -> ok`: `13`
- `ok -> ok`: `63`
- `missing_final_answer -> missing_final_answer`: `132`
- `ok -> missing_final_answer`: `35`
- `ok -> generation_error`: `10`

Interpretation:

- The oracle first-query fix often helps the agent escape early failure modes and actually reach a final answer.
- The largest positive structural gain is conversion from `missing_final_answer` to `ok`.
- The intervention is helpful but not universally stabilizing. Some previously successful cases regress, which means later turns can still derail even when the first query is repaired.

### GPTAcc Change Count

Across the 369-item subset:

- Improved: `32`
- Unchanged: `329`
- Regressed: `8`

This pattern is consistent with a targeted intervention:

- most samples do not change under a strict binary judge
- a meaningful minority improve
- only a small set regress

## Qualitative Findings

### Representative Improvements

Examples where first-round grounding clearly helps:

- `sensitivity to bright light causes` -> `albinism causes sensitivity to bright light`
- `river in the image with a building across from it` -> `Arkansas River`
- `architectural style of the building in the image` -> `New York Stock Exchange building`

These cases share the same pattern:

- the original query contains an unresolved visual referent
- the oracle rewrite converts it into a named entity or grounded concept
- retrieval becomes much more answerable from text

### Representative Regressions

Examples with negative `GPTAcc` delta:

- `music festivals in upstate NY named after a bird` -> `Great Blue Heron Music Festival`
- `What does the top line on the sign on the wall say in English?` -> `orden móvil y pago translation to English`
- `car seating capacity identification` -> `Toyota Venza`

These regressions suggest two remaining risks:

- the oracle rewrite may still be too narrow or slightly misaligned with the actual answer need
- even with a corrected first query, later agent turns can wander away from the target or fail to finalize correctly

## Interpretation

This experiment supports a strong causal conclusion:

- First-round entity ambiguity is a real bottleneck in the current multimodal agentic retrieval pipeline.
- Fixing only the first query already yields clear gains in answer quality.
- Therefore, the current performance ceiling is not determined only by retriever quality or later reasoning quality; it is also constrained by early query grounding failure.

At the same time, the gain is still limited:

- `GPTAcc` remains only `0.1165` after the oracle fix.
- Many cases still stay in `missing_final_answer`.
- Some successful baseline cases regress after the intervention.

So the first-query ambiguity problem is important, but it is not the only bottleneck.

## Conclusion

The experiment validates the central hypothesis behind entity-aware query repair:

- unresolved first-round queries materially hurt downstream RAG performance
- explicit entity grounding before the first retrieval step improves both retrieval-aligned metrics and judge-based answer correctness

Practical implication:

- A production fix should not rely on full oracle supervision
- but should introduce a lightweight first-round query diagnosis and grounding module
- and should likely be paired with later-turn stabilization to avoid regressions after the initial repair

## Next Step

The most direct follow-up is:

1. Replace the oracle rewrite with an automatic rewrite module.
2. Apply it only when the first query is predicted to be entity-ambiguous.
3. Compare:
   - baseline
   - first-round automatic rewrite
   - first-round oracle rewrite
4. Measure not only final metrics, but also:
   - retrieval hit quality
   - final-answer rate
   - loop length
   - repeated-query reduction

That experiment would tell us how much of the oracle gain is actually recoverable by a deployable method.
