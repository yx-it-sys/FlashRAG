# Entity Ambiguity Frequency Report

| Model | Label files | Ambiguous / Total | Frequency |
|---|---:|---:|---:|
| InternVL3.5-8B | 4 | 2090 / 5117 | 40.84% |
| Qwen2.5-vl-7B | 4 | 2685 / 8561 | 31.36% |
| Qwen3-vl-32B | 4 | 288 / 6795 | 4.24% |
| Qwen3-vl-2B | 0 | NA | NA |
| Qwen3-vl-4B | 4 | 1714 / 8638 | 19.84% |
| Qwen3-vl-8B | 4 | 1585 / 8425 | 18.81% |

Notes:
- `Qwen3-vl-2B` has no `omnisearch_trajectories.entity_ambiguity_labeled.jsonl` files in this checkout, so its frequency cannot be computed from the requested labels.
- Frequencies are computed as `#(entity_ambiguous == Yes) / #(all labeled steps)` across the base four source runs for each model.
