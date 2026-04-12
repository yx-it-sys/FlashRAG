# Entity Ambiguity Category Metrics

- Samples: 1199
- Retained samples: 1131
- Filtered multi-category samples: 68
- Category rule: Exclude samples containing more than one non-No entity ambiguity category across labeled query steps; otherwise, use the first LLM-labeled text-retrieval query category in the sample. If no labeled text query exists, assign No.
- Iteration rule: Count retrieval-result turns in omnisearch_trajectories.jsonl, i.e. the number of text_retrieval_result, image_retrieval_result, and no_retrieval_result nodes.

| Category | Count | Avg Iterations | Avg Query Count | GPTAcc | F1 | Recall | Precision | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No | 824 | 3.011 | 2.461 | 0.277 | 0.284 | 0.438 | 0.259 | 0.213 |
| Object Identification | 183 | 4.115 | 2.776 | 0.049 | 0.150 | 0.270 | 0.127 | 0.147 |
| Indirect Entity Ambiguity | 56 | 3.661 | 3.482 | 0.036 | 0.186 | 0.344 | 0.158 | 0.166 |
| Description | 64 | 3.266 | 2.625 | 0.094 | 0.211 | 0.328 | 0.207 | 0.183 |
| No Object Involved | 4 | 2.750 | 2.000 | 0.250 | 0.316 | 0.519 | 0.255 | 0.125 |
