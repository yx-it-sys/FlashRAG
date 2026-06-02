# First-Round Disturb Rewrite: Mechanism Cases

## Goal

This note keeps only the cases that explain why a blurred first-round query can improve generation quality, even when the retrieval result looks weak.

The mechanism is not that the noisy retrieval itself contains the answer. The mechanism is that the blurred query changes the model's search state:

1. it weakens premature entity locking,
2. it forces the model to recover an intermediate entity,
3. it makes later queries more specific,
4. it creates a guess-verify-revise loop,
5. it lets visual/contextual priors dominate when text retrieval is unhelpful.

All cases below satisfy the stricter filter:

- disturbed trajectory ends in `ok`
- baseline is not `ok`
- `gpt_acc == 1.0`

## Case 1: Weak retrieval still helps because it triggers a visual re-anchor

### Case: `refamb_001230`

- Source stage: `RefAmb_2026_05_02_13_50_refamb_infoseek_stage`
- Baseline -> disturbed: `missing_final_answer -> ok`
- Original query: `Which continent is North America part of?`
- Disturbed first query: `Which continent is this region part of?`

Trajectory:

```text
Identify the mountain in the image.
Which continent is this region part of?
```

What happens:

- The first text retrieval result is noisy and does not directly answer the question.
- The model does not rely on retrieval alone.
- It uses the image to infer the mountain identity first, then maps that mountain to Alaska and North America.

Why this supports the point:

- A weak query can still be useful because it creates a mismatch signal.
- That mismatch pushes the model to search the visual input for a stable intermediate anchor.
- The intermediate entity is not extracted from retrieval; it is recovered from the VLM's visual prior.

## Case 2: The blurred query forces intermediate-entity recovery before the final target

### Case: `refamb_000659`

- Source stage: `RefAmb_2026_05_02_14_22_refamb_crag_stage`
- Baseline -> disturbed: `generation_error -> ok`
- Original query: `capital of Bavaria`
- Disturbed first query: `capital of that place`

Trajectory:

```text
capital of that place
largest stadium in Munich
```

What happens:

- The first query is too generic to stay locked onto Bavaria as a named entity.
- The model first recovers the intermediate city: `Munich`.
- Only after that does it ask the downstream query about the largest stadium.

Why this supports the point:

- Disturb removes the direct shortcut to the final target.
- That forces the model to reconstruct the middle node of the chain.
- Once the middle node is found, the final query becomes much more precise.

## Case 3: Retrieval mismatch makes the model re-plan the trajectory

### Case: `refamb_000806`

- Source stage: `RefAmb_2026_05_02_14_22_refamb_crag_stage`
- Baseline -> disturbed: `generation_error -> ok`
- Original query: `calcium content of broccoli and cucumbers`
- Disturbed first query: `calcium content of these vegetables`

Trajectory:

```text
calcium content of these vegetables
calcium content of broccoli and cucumbers
No Retrieval
```

What happens:

- The first search is generic and not directly informative.
- The model reacts by restating the full comparison pair.
- After that, the answer becomes a clean binary comparison: broccoli has more calcium.

Why this supports the point:

- The weak first retrieval is not the end of the process.
- It acts like a failed probe that tells the model the query is still underspecified.
- That failure causes the model to re-plan with the full entity pair.

## Case 4: Disturb prevents early lock-in to a too-specific but wrong path

### Case: `refamb_002819`

- Source stage: `RefAmb_2026_05_01_13_28_refamb_oven_stage`
- Baseline -> disturbed: `missing_final_answer -> ok`
- Original query: `toilet flush button function`
- Disturbed first query: `button function for flushing`

Trajectory:

```text
button function for flushing
```

What happens:

- The original query is already close to the answer, but not enough to force a robust reconstruction.
- The disturbed query generalizes the action first.
- The model then cleanly lands on the intended function of the toilet button.

Why this supports the point:

- If the model is locked too early onto one surface form, it may produce an incomplete trajectory.
- The blur makes it rebuild the action in a more abstract way first.
- That abstraction is easier to ground consistently in the image and the final answer.

## Case 5: The real pattern is guess -> verify -> revise

### Combined evidence: `refamb_001230`, `refamb_000659`, `refamb_000806`

These cases all show the same control flow:

1. the model makes an initial guess from image/context,
2. the generic first query fails to settle the exact entity,
3. the model revises the query to a more concrete entity,
4. the final answer becomes correct.

This is visible in:

- `refamb_001230`: mountain -> North America
- `refamb_000659`: place -> Munich
- `refamb_000806`: vegetables -> broccoli and cucumbers

Why this matters:

- Disturb does not need the first retrieval to be informative.
- Its job is to expose ambiguity so the model can self-correct.
- The model uses that ambiguity to move from a guess to a verified intermediate anchor, and then to a precise final query.

## Takeaway

The blurred first query improves generation quality not because the retrieved text is directly useful, but because it changes the reasoning dynamics:

- it suppresses over-specific early commitments,
- it pushes the model to recover a better intermediate entity,
- it encourages a more precise second query,
- and it lets the VLM use visual/contextual priors to bridge weak retrieval.

That is why even low-quality retrieval can still lead to better final answers.
