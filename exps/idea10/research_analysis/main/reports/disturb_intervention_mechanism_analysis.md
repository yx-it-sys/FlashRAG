# Disturb Intervention Mechanism Analysis

This note analyzes the first-round Disturb intervention across four RefAmb sources:
MCSearch, OVEN, InfoSeek, and CRAG. The key question is why replacing an originally
unambiguous first text query with a more ambiguous or generic query can sometimes improve
trajectory outcomes.

## Main Conclusion

Disturb does not improve performance by making retrieval more semantically accurate.
Instead, it acts as a stochastic relaxation of the first retrieval constraint. This relaxation
increases factual novelty and sometimes helps trajectories escape early query stagnation,
especially when the original query over-commits to a fragile entity name or when the task
requires visual-to-text referent transfer.

However, because the relaxed query weakens entity anchoring, the additional evidence is often
less usable. The dominant pattern is therefore: Delta F increases, but utility and answer
quality become unstable.

## Raw Aggregate Table

| Source | n | Main status shifts | Delta TQS | Delta total Delta F | Delta iter | Delta utility | Delta F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CRAG | 179 | 28 fail->OK, 25 OK->fail | -0.0667 | +0.4093 | +0.4972 | -0.1268 | -0.0226 |
| InfoSeek | 26 | 6 fail->OK, 3 OK->fail | -0.0649 | +0.5565 | -0.0385 | -0.1538 | +0.0261 |
| MCSearch | 60 | 4 fail->OK, 10 OK->fail | -0.0132 | +0.5569 | +0.4667 | -0.0882 | -0.0476 |
| OVEN | 35 | 5 fail->OK, 4 OK->fail | +0.0483 | +0.2401 | -0.4571 | -0.0495 | +0.0143 |

Across all 300 Disturb items:

| Transition | n | Source composition | Delta TQS | Delta total Delta F | Delta iter | Delta utility | Delta F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| generation_error->OK | 16 | CRAG 16 | +0.3587 | -0.0222 | -1.4375 | +0.2222 | +0.1556 |
| missing_final_answer->OK | 27 | CRAG 12, InfoSeek 6, OVEN 5, MCSearch 4 | +0.3205 | +0.3169 | -2.5926 | +0.0894 | +0.1490 |
| OK->missing_final_answer | 23 | MCSearch 10, CRAG 6, OVEN 4, InfoSeek 3 | -0.2997 | +0.8675 | +2.8696 | -0.3444 | -0.1832 |
| OK->generation_error | 19 | CRAG 19 | -0.4278 | +0.5957 | +2.6316 | -0.4306 | -0.3399 |
| OK->OK | 110 | CRAG 53, MCSearch 37, OVEN 14, InfoSeek 6 | -0.0976 | +0.3658 | +0.3455 | -0.1495 | -0.0148 |
| failure->failure | 105 | all sources | near flat | positive | mixed | lower | near flat |

The most important aggregate pattern is that Disturb increases Delta F for most cohorts
while decreasing utility for most cohorts. This supports the interpretation that Disturb is
an exploration intervention, not an alignment intervention.

## Sample Type Analysis

Recovered samples are concentrated in visually grounded, external-knowledge-dependent cases.
Among the 43 failure-to-OK recovered items:

| Attribute | Count |
| --- | ---: |
| text_alone_identifies_entity = No | 38 |
| text_alone_identifies_entity = Yes | 5 |
| external_evidence_dependency = Yes | 40 |
| external_evidence_dependency = No | 3 |

Task-type distribution of recovered samples:

| Task type | Recovered count |
| --- | ---: |
| Single-hop Attribute Query | 12 |
| Entity Recognition | 10 |
| Multi-hop | 10 |
| Comparison | 7 |
| Subproblem Aggregation | 4 |

By contrast, OK-to-failure cases are also mostly external-knowledge-dependent. This means the
same relaxation can either recover a stalled trajectory or destroy a previously useful entity
anchor.

## Mechanism

### 1. Constraint Relaxation

Disturb often replaces an entity name with a deictic or category-level phrase:

| Original query | Disturbed query |
| --- | --- |
| `jaguar gestation period` | `big cat gestation period` |
| `license plate format M996 JPA` | `license plate format of that vehicle` |
| `Roman collection at the Yorkshire Museum` | `Their collection at that museum` |
| `typical number of strings for a baroque lute` | `typical number of strings for that instrument` |

This weakens exact entity anchoring, but it can make the query better match generic knowledge
patterns in the corpus. In such cases, the first retrieval returns broader background facts
that help the agent reformulate a better second query.

### 2. Escaping Local Stagnation

In several recovered trajectories, the first disturbed query is not itself ideal. Its value is
that it changes the trajectory state. The agent then issues a more specific follow-up query.
For example:

`Their collection at that museum` -> `Roman collection at the Yorkshire Museum artifacts`

The disturbed query therefore behaves like a perturbation that moves the agent away from a
stale query path. This is why failure-to-OK transitions show fewer iterations on average
(-2.5926 for missing_final_answer->OK) while answer quality improves.

### 3. Visual-to-Text Referent Transfer

Recovered cases are often visually grounded: the entity is available in the image, but the
first text query is a fragile textual projection of that entity. A generic phrase such as
`this animal`, `that vehicle`, or `that instrument` can keep the agent's attention on the
visual referent and induce later retrieval that better matches the question's attribute.

### 4. Topic Drift

The downside is severe. OK-to-failure transitions have strongly positive Delta F but sharply
negative utility and F1. This means the agent retrieves many new facts, but the facts are
about the wrong entity, wrong type, or wrong branch of a multi-part question.

Example drift patterns:

| Original query | Disturbed query | Failure mode |
| --- | --- | --- |
| `common names for Hydrangea quercifolia` | `common names for this plant` | retrieves facts about a different plant |
| `Cadillac Series 62 bodystyles...` | `That luxury car model's bodystyles...` | drifts to Lamborghini/Aventador facts |
| `location of the Spice Bazaar` | `location of the well-known marketplace` | retrieves another marketplace |

This explains why Disturb can increase total Delta F while decreasing final-answer quality.

## Paper-Ready Claim

Disturb should be described as an exploratory relaxation rather than a robust improvement.
The paper should state:

> Disturb increases trajectory vitality by broadening the first retrieval constraint, which
> raises factual novelty and sometimes helps stalled trajectories recover. Yet the same
> relaxation weakens entity anchoring, so the additional facts are often less usable and can
> trigger topic drift. This explains the non-monotonic effect of referential ambiguity: ambiguity
> can be useful as exploration, but reliable answer improvement requires returning to the golden
> entity.

## Suggested Follow-up Experiments

1. Measure how often a recovered Disturb trajectory returns to a more specific entity query in
   the second or third retrieval step.
2. Plot Delta F and utility jointly for Disturb, Oracle, and Original to show that Disturb is
   DeltaF-up/utility-down, while Oracle should be DeltaF-up/utility-up.
3. Label OK-to-failure cases by drift type: wrong entity, wrong category, wrong subproblem,
   or context-window degeneration.
4. Add a controlled hybrid method: allow Disturb-like exploration only when a trajectory has
   already shown low gain, then force a golden-entity-aligned rewrite after the exploratory step.
