# Introduction Figure Case (RefAmb)

- Case ID: `refamb_000873`
- Source result file: `/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/merged/omnisearch_trajectories.jsonl`
- Status: `ok`
- Question: `What is the amount (percent) of dissolved salt in this lake?`
- Final Answer: `The percentage of dissolved salt in the Dead Sea is 34.2%.`

## ReAct Core Flow

1. **Thought**: Need to identify which lake appears in the image before answering salinity.
2. **Search (Text Retrieval)**: query the lake identity; retrieved text is noisy/irrelevant.
3. **Thought**: Text evidence is insufficient; switch retrieval modality.
4. **Search (Image Retrieval)**: retrieves `Dead Sea` as top entity.
5. **Thought**: Entity resolved as `Dead Sea`.
6. **Search (Text Retrieval)**: query salinity of Dead Sea.
7. **Thought**: retrieved evidence states salinity is `34.2%`.
8. **Final Answer**: output `34.2%`.

## Why This Case Fits Introduction

- Demonstrates **referential ambiguity**: "this lake" is underspecified without visual grounding.
- Demonstrates **cross-modal disambiguation**: text retrieval fails first, image retrieval resolves the entity.
- Demonstrates **evidence chaining**: entity identification -> attribute lookup -> final answer.
