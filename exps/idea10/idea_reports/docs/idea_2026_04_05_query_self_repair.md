# Idea Note: Query Self-Repair Failure in Agentic Retrieval

## Core Problem

In the current agentic pipeline, the generated retrieval query lacks self-repair ability. Once the first query is off-target, the later iterations often stay trapped around the same query pattern instead of actively correcting it. As a result, the whole search process keeps circling around nearly identical queries and fails to gather new evidence.

## Current Observation

A concrete issue is that the generated query frequently contains references or pronouns such as "this building", "the place", or "the object in the image". However, the retriever itself cannot see the image. It only receives the text query.

This creates a modality mismatch:

- The generator implicitly relies on visual context.
- The retriever only sees unresolved text.
- The core entity in the image is therefore missing from the query.
- Retrieval recall becomes weak because the query does not name the real target object.

## Failure Mechanism

The failure can be described as a simple loop:

1. The model observes an image and writes a query with unresolved reference.
2. The retriever cannot recover the intended visual referent from text alone.
3. Retrieved evidence is weak or off-target.
4. The agent does not truly rewrite the query with a grounded object name.
5. The next round repeats the same search intent with minor surface variation.

This means the bottleneck is not only retrieval quality itself, but the lack of query grounding and query self-repair inside the agent loop.

## Hypothesis

If the pipeline can explicitly detect unresolved references in generated queries and force a grounded rewrite before retrieval, recall should improve. In particular, the query should be rewritten from image-dependent reference form into object-explicit textual form.

Example:

- Weak query: "What events are held at this place?"
- Better query: "What events are held at Civic Center Plaza?"

## Tentative I
For "Object Identification" problem:
1. Annotate ROI of Image Query based on the original text query, then clip into an entity-focus image. 
2. Then image retrieval and return top-k results
3. If results contain various entitis, let VLM decide which entity to choose while computing confusion score (uncertainty) of VLM for each candidates?
4. If not obvious: Scene Description
    1. Describe the surrounding environment where the entity in and formulating clues.
    2. generate several query probs and retrieve
    3. build a KG of retrieval results
    4. pruning and flatten
    5. generate a detailed descriptionof entity
5. Reformulating query