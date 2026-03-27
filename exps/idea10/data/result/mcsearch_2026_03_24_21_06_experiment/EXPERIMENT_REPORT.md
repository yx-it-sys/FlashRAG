# MCSearch Experiment Report

## Experiment Overview

- Experiment directory: `data/result/mcsearch_2026_03_24_21_06_experiment`
- Primary trajectory file: `omnisearch_trajectories.jsonl`
- Intermediate process file: `intermediate_data.json`
- Aggregate metrics from `metric_score.txt`:
  - EM: `0.0`
  - F1: `0.4129`
  - Acc: `0.005`
  - Precision: `0.3915`
  - Recall: `0.4920`
- Timing from `records.txt`:
  - Total: `2378.21s`
  - Count: `200`
  - Avg/item: `11.89s`

## ReAct Case Study

This section analyzes the ReAct process using:

- `omnisearch_trajectories.jsonl` for online execution traces
- `intermediate_data.json` for graph type annotations, sub-question chains, and per-sample metric scores

### Process Statistics

| Metric | Value |
|---|---:|
| Number of samples | 200 |
| `ok` | 144 |
| `missing_final_answer` | 56 |
| Mean trajectory length | 15.1 steps |
| Median trajectory length | 14 steps |
| Mean duration | 11.86s |
| Mean sub-question count | 3.81 |

### Graph Type Breakdown

| Graph Type | Total | `ok` | `missing_final_answer` | Mean F1 |
|---|---:|---:|---:|---:|
| `Image-Initiated Chain` | 91 | 62 | 29 | 0.428 |
| `Parallel Visual-Textual Fork` | 54 | 37 | 17 | 0.401 |
| `Text Chain` | 45 | 38 | 7 | 0.425 |
| `Text-Initiated Chain` | 10 | 7 | 3 | 0.287 |

### Core Observation

The main bottleneck is not purely sub-question decomposition. A large fraction of failures are process-control failures:

- Some samples already contain a nearly complete evidence chain, but never trigger final answer synthesis.
- Some samples enter repeated retrieval loops after the first failed search and never repair the query.

This is visible in two aggregate signals:

- Mean F1 for `missing_final_answer` is still `0.182`, so these samples are not uniformly hopeless failures.
- Failed trajectories have much higher repeated-search behavior:
  - `ok`: mean repeat ratio `0.033`
  - `missing_final_answer`: mean repeat ratio `0.637`

### Case 1: Successful Short-Chain ReAct

Question:

> Given the image displays an art gallery, which university's art collection has been housed in an art gallery since 1975, and what specific aspect makes that gallery notable?

Metadata:

- Graph type: `Image-Initiated Chain`
- Status: `ok`
- F1: `0.960`
- Steps: `10`

Gold answer:

> New York University's art collection has been housed in the Grey Art Gallery since 1975, and it is notable for its museum-quality exhibitions of contemporary art.

Trajectory pattern:

1. Identify image as an art gallery
2. Ask which university collection has been housed in a gallery since 1975
3. Retrieve `New York University` and `Grey Art Gallery`
4. Ask what makes Grey Art Gallery notable
5. Retrieve `museum-quality exhibitions of contemporary art`
6. Synthesize final answer

Why this case works:

- The image provides a coarse semantic anchor rather than a brittle entity guess.
- Each sub-question introduces new evidence.
- The system stops once the answer slots are filled instead of over-searching.

This is a strong example of multimodal ReAct working as intended: image for category grounding, text for entity binding and attribute completion.

### Case 2: High-Quality Failure Caused by Missing Final Synthesis

Question:

> Considering the public space depicted in the image that hosts various events, what types of large public gatherings take place there, and what categories of significant institutions are found in the broader Civic Center area, including a specific judicial body located east on Mission Street?

Metadata:

- Graph type: `Image-Initiated Chain`
- Status: `missing_final_answer`
- F1: `0.323`
- Steps: `23`

The `subqa_chain` already contains nearly all required evidence:

- `Civic Center Plaza`
- `street parades and parties`, including `Gay Pride Parade` and `Earth Day celebration`
- `government and cultural institutions`
- `head courthouse of the United States Court of Appeals for the Ninth Circuit`

However, the online trajectory repeatedly asks the same sub-question and issues near-duplicate searches, then stops without producing a final answer.

Interpretation:

- The reasoning graph is largely correct.
- The failure is in stopping and synthesis, not necessarily in retrieval coverage.

Implication:

This is evidence that process supervision should target answer-readiness detection, not only better search.

### Case 3: Failure Caused by Retrieval Drift and No Query Repair

Question:

> What artistic field is promoted by the non-profit media arts center that is also located in the same downtown area as the Cultural Arts Center, Montgomery College, pictured in the image?

Metadata:

- Graph type: `Image-Initiated Chain`
- Status: `missing_final_answer`
- F1: `0.000`
- Steps: `23`

The oracle-style `subqa_chain` in `intermediate_data.json` is clean:

- `Cultural Arts Center, Montgomery College`
- `Downtown Silver Spring`
- `Docs in Progress`
- `Documentary filmmaking`

But the actual ReAct trajectory retrieves an irrelevant `National Music Centre` result and then repeats essentially the same failed query multiple times, never shifting strategy.

Interpretation:

- The issue is not decomposition quality.
- The issue is lack of retrieval self-repair after an off-target hit.

Implication:

Loop prevention and adaptive query rewriting are likely higher-leverage than simply increasing retrieval depth.

## Failure Mode Summary

The collected evidence supports a two-way failure taxonomy:

1. Synthesis failure
   - Evidence chain is mostly complete
   - The system does not switch from search mode to answer mode
   - Typical symptom: long trajectory ending in search or thought without `final_answer`

2. Retrieval-loop failure
   - First retrieval is wrong or too generic
   - The system repeats the same query with minimal reformulation
   - Typical symptom: high search repetition ratio and no new evidence gain

There is also a logging inconsistency worth noting:

- `status=ok` samples: 144
- samples with explicit `final_answer` action: 117

This suggests some successful outputs may be recorded without a distinct `final_answer` action, so future analysis should distinguish answer text presence from action label presence.

## Potential Innovation Points

### 1. Answer-Readiness Controller

Introduce an explicit controller that decides when enough evidence has been collected to stop retrieval and synthesize the final answer.

Why it matters:

- Directly targets the `missing_final_answer` class
- Converts partially successful trajectories into completed answers

### 2. Loop-Aware ReAct

Track repeated or semantically equivalent search queries and penalize redundant retrieval loops.

Possible behaviors after repeated failure:

- force query rewriting
- switch graph branch
- synthesize with current evidence
- abstain if coverage remains low

Why it matters:

- Failed samples show dramatically higher repeated-query ratios
- This is a process-control problem that standard ReAct does not explicitly handle

### 3. Graph-Supervised ReAct

Use `graph_type` and `subqa_chain` from `intermediate_data.json` as process supervision for a plan-then-execute ReAct variant.

Why it matters:

- The dataset already contains latent high-quality reasoning graphs
- Online execution is often worse than the offline chain annotation
- This creates a natural training signal for structured reasoning control

### 4. Evidence-Coverage Verifier

Represent the question as slots such as:

- target entity
- location
- time
- attribute or relation

Then track whether each slot has supporting evidence. Retrieval should be directed toward uncovered slots, and synthesis should trigger when required coverage is satisfied.

Why it matters:

- Prevents unnecessary extra search once the answer is already recoverable
- Makes stopping interpretable and measurable

### 5. Failure-Type Adaptive Retrieval

Dynamically switch retrieval behavior based on online failure signals:

- wrong entity hit: strengthen entity constraints
- generic document hit: strengthen attribute constraints
- repeated no-gain search: branch to another graph pattern

Why it matters:

- Different failures require different interventions
- A static retrieval policy is too brittle for multimodal multi-hop QA

### 6. Process Reward Instead of Final-Only Reward

Optimize for intermediate process quality, not only final answer metrics.

Candidate rewards:

- evidence gain per step
- sub-question completion rate
- query diversity
- slot coverage
- successful termination

Why it matters:

- `ok` is not always equivalent to fully correct
- `missing_final_answer` is not always equivalent to reasoning failure
- Final-only supervision hides actionable process differences

## Method Motivation

The strongest takeaway from this case study is:

> The primary bottleneck of multimodal ReAct in this experiment is process control rather than raw decomposition ability.

More specifically:

- the system often knows how to decompose the problem
- the system sometimes even gathers enough evidence
- but it does not robustly decide when to stop, repair, or synthesize

This motivates a shift:

> from free-running ReAct to process-controlled ReAct with explicit graph supervision and evidence coverage tracking

## Recommended Next Experiments

1. Add a simple repeated-query detector and force query rewrite after two near-duplicate searches.
2. Add an answer-readiness heuristic based on sub-question completion or slot coverage.
3. Compare vanilla ReAct against graph-supervised ReAct using the existing `subqa_chain` annotations.
4. Report not only EM/F1, but also:
   - completion rate
   - repeated-query ratio
   - average unique queries per sample
   - final-synthesis success rate
