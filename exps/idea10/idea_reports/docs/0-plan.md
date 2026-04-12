# Annotation Plan

## Task

Annotate whether each text retrieval query generated during the RAG trajectory contains referential wording that makes the target entity ambiguous for a text-only retriever, and assign one of five severity levels.

The working annotation file is:

- `/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/trajectory_annotation.jsonl`

The manual annotation subset is:

- `/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/trajectory_annotation_manual_sample.jsonl`

## Annotation Unit

Each record corresponds to one test sample.

Under each record, `query_steps` contains all text retrieval queries generated during the RAG process for that sample. Each query step should be annotated independently.

## Sampling Plan

The manual annotation subset is stratified by the number of text retrieval queries (`query_count`) per test sample, instead of pure random sampling.

Current manual subset size:

- 120 test samples

Stratified distribution:

- `query_count = 1`: 39
- `query_count = 2`: 14
- `query_count = 3`: 9
- `query_count = 4`: 10
- `query_count = 5`: 48

Samples with `query_count = 0` are excluded from manual annotation because they do not contain any text retrieval query to label.

## Annotation Instructions
For each query, sign "Yes" or "No", default as "No" if you don't give a sign.
There are five levels of entity ambiguouty:
- Level 1 Description: No definite entity name but offers a descriptive constraint phrases or detailed / useful categorization.
- Level 2 Object Identification: The query realizes the neccessity to recognize the entity but give this task to search engine. (Identify ... in the image)
- Level 3 Normal enitity ambiguouty: have pronouns and no affiliated useful infomation (too general categorization). (eg. Where is the sculpture located?)
- Level 4 No Object shown: there is no words standing for the entity at all.
## Annotator Setup

Recommended setup for paper-quality evidence:

- 2 independent annotators for the same 120-sample manual subset
- 1 optional adjudicator for disagreement resolution

Practical assignment:

- Annotator A can be the author
- Annotator B should be another independent person familiar with the task
- If disagreements remain, use a third person or discussion-based adjudication to create final labels

## Agreement Evaluation

Use the 120-sample manual subset for inter-annotator agreement analysis.

Recommended reporting:

- compute Cohen's kappa on the two independent annotations
- report the final adjudicated labels separately from raw agreement
- state clearly that the subset was stratified by query-count complexity

## Notes

- A single annotator is acceptable for pilot exploration, but not strong enough for a top-tier paper's formal agreement claim.
- For the formal paper result, the main evidence should be based on dual annotation with explicit agreement statistics.

## Detector Goal

Train an entity ambiguity detector that monitors each generated text retrieval query during the RAG trajectory.

When the detector predicts that a query is entity-ambiguous, the system should trigger a warning and route the trajectory into an entity-verification or query-rewrite stage before continuing retrieval.

This detector is intended to act as a lightweight and stable trajectory guardrail rather than replacing the main LLM.

## Detector Task Definition

Recommended formulation:

1. Binary classification
- Input: `question + sub_question + text_query`
- Output: `ambiguous` or `not_ambiguous`

2. Positive-only subtype classification
- Input: the same query-step context, only for binary-positive samples
- Output:
  - `Object Identification`
  - `Description`
  - `Indirect Entity Ambiguity`
  - `No Object Involved`

This two-stage design is preferred over a single multi-class model for deployment, because online triggering only depends on the binary detector and the subtype classifier can be used for analysis or routing.

## Training Data Sources

The detector should be trained at the query-step level, not the sample level.

Each `query_steps[i]` in `trajectory_annotation.llm_labeled.jsonl` is one training instance.

Recommended data sources:

1. Gold labels
- manually reviewed query-step labels from the annotation subset
- use these primarily for development and test evaluation

2. Silver labels
- LLM-generated ambiguity labels from:
- `/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/trajectory_annotation.llm_labeled.jsonl`
- use these to enlarge the training set

3. Contrastive rewrite pairs
- original ambiguous queries as positive examples
- corresponding human-rewritten or LLM-rewritten explicit queries as hard negative examples
- these pairs are especially useful for learning the ambiguity boundary

Recommended per-instance fields:

- `annotation_id`
- `step_index`
- `question`
- `sub_question`
- `text_query`
- `rewrite_query` if available
- `label_binary`
- `label_type`
- `status`
- `query_count`
- `source` such as `gold`, `silver`, or `rewrite_negative`

## Data Construction Principles

Positive samples:

- query steps with `llm_label.entity_ambiguous == "Yes"`
- prioritize manually validated positives when available

Negative samples:

- query steps with `llm_label.entity_ambiguous == "No"`
- rewritten versions of ambiguous queries
- manually validated no-rewrite cases

Important split rule:

- split train, development, and test by `annotation_id`, not by individual query step
- this avoids leakage from repeated or nearly duplicated query templates across the same sample

## Recommended Learning Method

Start with supervised classification rather than generative modeling.

Recommended training order:

1. Binary detector
- supervised classification for `ambiguous` vs `not_ambiguous`

2. Subtype classifier
- supervised classification on the positive subset only

3. Optional later extensions
- contrastive learning using original-vs-rewritten query pairs
- teacher-student distillation from an LLM judge
- confidence calibration for trigger thresholds

This is preferable to directly fine-tuning a large instruction model for the detector.

## Recommended Models

Strong practical recommendation:

1. Lexical baseline
- TF-IDF + Logistic Regression

2. Main detector
- `DeBERTa-v3-base` or another encoder-style classifier of similar scale

3. Optional embedding-based baseline
- `e5-base-v2` or `bge-base` encoder + MLP head

The main reason to prefer encoder classifiers:

- this is a short-text discrimination task rather than a long-form reasoning task
- encoder models are cheaper, faster, easier to calibrate, and better suited for online triggering

Large instruction-tuned LLMs should mainly be used as:

- silver-label generators
- teacher models
- rewrite modules

not as the first-choice online detector.

## Detector Input Format

Recommended textual input template:

```text
[Question]
...

[SubQuestion]
...

[TextQuery]
...
```

Optional metadata features such as `step_index`, `query_count`, and `status` can be added later if needed, but the first version should stay text-only.

## Online Trigger Logic

Recommended deployment policy:

- if detector score < `t1`, continue normal retrieval
- if `t1 <= score < t2`, record a low-confidence warning
- if detector score >= `t2`, trigger entity verification or query rewriting

This thresholded design is better than a single hard decision because it supports precision-recall tradeoff tuning for different system goals.

## Validation Plan For Top-Tier Evidence

The detector should not be evaluated only by classification accuracy.

A top-tier validation design should include three layers:

1. Rewrite correctness
- manually inspect whether the rewritten query truly resolves ambiguity
- recommended labels:
  - `entity_resolved`
  - `intent_preserved`
  - `retrieval_ready`
  - `overall_judgment`

2. Retrieval effect
- compare original ambiguous queries against rewritten queries
- recommended metrics:
  - Recall@k
  - MRR@k
  - nDCG@k
  - answer-hit@k if answer-containing evidence is available

3. End-task effect
- compare end-to-end QA performance with:
  - no detector
  - always rewrite
  - oracle trigger
  - learned detector trigger

## Core Experimental Tables

Recommended paper-ready reporting:

1. Detector classification table
- Precision / Recall / F1 / AUROC / AUPRC

2. Retrieval comparison table
- original query vs rewritten query

3. End-to-end QA comparison table
- no detector vs always rewrite vs oracle trigger vs trained detector trigger

4. Ambiguity-type breakdown table
- `Object Identification`
- `Description`
- `Indirect Entity Ambiguity`
- `No Object Involved`

5. Error analysis table
- wrong entity
- under-specific rewrite
- over-specific rewrite
- answer leakage
- malformed query
- no rewrite needed

## Key Risk

The largest paper risk is answer leakage.

Because some rewrite experiments use gold `Answers` to identify the target entity, such settings should be explicitly described as oracle-guided or answer-grounded query clarification.

This is useful for measuring whether entity ambiguity is a real bottleneck, but it should not be presented as a fully deployable setting without an additional practical version that uses model-predicted entity grounding instead of gold answers.
