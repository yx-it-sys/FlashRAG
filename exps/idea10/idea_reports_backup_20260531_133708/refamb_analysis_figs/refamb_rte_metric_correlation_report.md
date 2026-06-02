# RTE vs Generation-Metric Correlation

- Samples: 1500
- Sources: {'crag': 818, 'infoseek': 176, 'mcsearch': 300, 'oven': 206}
- Task types: {'Comparison': 300, 'Entity Recognition': 300, 'Multi-hop': 300, 'Single-hop Attribute Query': 300, 'Subproblem Aggregation': 300}
- Status counts: {'generation_error': 325, 'missing_final_answer': 420, 'ok': 755}

## Overall Correlations

| Metric | n | Pearson r | Pearson p | Spearman rho | Spearman p |
| --- | ---: | ---: | ---: | ---: | ---: |
| EM | 1500 | 0.0447 | 0.0836 | 0.0309 | 0.2318 |
| Acc | 1500 | -0.1050 | 0.0000 | -0.1431 | 0.0000 |
| F1 | 1500 | 0.5385 | 0.0000 | 0.5255 | 0.0000 |
| Recall | 1500 | 0.1965 | 0.0000 | 0.2285 | 0.0000 |
| Precision | 1500 | 0.5649 | 0.0000 | 0.5582 | 0.0000 |
| GPTAcc | 1500 | 0.4059 | 0.0000 | 0.3619 | 0.0000 |

## Key Findings

1. `RTE` vs `Precision`: Spearman rho = 0.5582, Pearson r = 0.5649, n = 1500.
1. `RTE` vs `F1`: Spearman rho = 0.5255, Pearson r = 0.5385, n = 1500.
1. `RTE` vs `GPTAcc`: Spearman rho = 0.3619, Pearson r = 0.4059, n = 1500.

## Stratified Correlations

| Group Field | Group Name | Metric | n | Spearman rho | Spearman p |
| --- | --- | --- | ---: | ---: | ---: |
| source | crag | EM | 818 | 0.0752 | 0.0314 |
| source | crag | Acc | 818 | 0.0752 | 0.0314 |
| source | crag | F1 | 818 | 0.4545 | 0.0000 |
| source | crag | Recall | 818 | 0.1993 | 0.0000 |
| source | crag | Precision | 818 | 0.4880 | 0.0000 |
| source | crag | GPTAcc | 818 | 0.3955 | 0.0000 |
| source | infoseek | EM | 176 | nan | nan |
| source | infoseek | Acc | 176 | 0.3368 | 0.0000 |
| source | infoseek | F1 | 176 | 0.2581 | 0.0005 |
| source | infoseek | Recall | 176 | 0.2809 | 0.0002 |
| source | infoseek | Precision | 176 | 0.2558 | 0.0006 |
| source | infoseek | GPTAcc | 176 | 0.3308 | 0.0000 |
| source | mcsearch | EM | 300 | -0.0925 | 0.1098 |
| source | mcsearch | Acc | 300 | -0.0962 | 0.0965 |
| source | mcsearch | F1 | 300 | 0.2808 | 0.0000 |
| source | mcsearch | Recall | 300 | 0.2187 | 0.0001 |
| source | mcsearch | Precision | 300 | 0.2739 | 0.0000 |
| source | mcsearch | GPTAcc | 300 | 0.2453 | 0.0000 |
| source | oven | EM | 206 | nan | nan |
| source | oven | Acc | 206 | 0.0020 | 0.9778 |
| source | oven | F1 | 206 | -0.0150 | 0.8305 |
| source | oven | Recall | 206 | 0.0391 | 0.5771 |
| source | oven | Precision | 206 | -0.0142 | 0.8400 |
| source | oven | GPTAcc | 206 | -0.0164 | 0.8148 |
| task_type | Comparison | EM | 300 | nan | nan |
| task_type | Comparison | Acc | 300 | nan | nan |
| task_type | Comparison | F1 | 300 | 0.4826 | 0.0000 |
| task_type | Comparison | Recall | 300 | 0.1600 | 0.0055 |
| task_type | Comparison | Precision | 300 | 0.5308 | 0.0000 |
| task_type | Comparison | GPTAcc | 300 | 0.4810 | 0.0000 |
| task_type | Entity Recognition | EM | 300 | nan | nan |
| task_type | Entity Recognition | Acc | 300 | -0.0757 | 0.1909 |
| task_type | Entity Recognition | F1 | 300 | 0.0638 | 0.2706 |
| task_type | Entity Recognition | Recall | 300 | 0.0376 | 0.5164 |
| task_type | Entity Recognition | Precision | 300 | 0.0764 | 0.1871 |
| task_type | Entity Recognition | GPTAcc | 300 | 0.0138 | 0.8114 |
| task_type | Multi-hop | EM | 300 | nan | nan |
| task_type | Multi-hop | Acc | 300 | nan | nan |
| task_type | Multi-hop | F1 | 300 | 0.5505 | 0.0000 |
| task_type | Multi-hop | Recall | 300 | 0.2844 | 0.0000 |
| task_type | Multi-hop | Precision | 300 | 0.5755 | 0.0000 |
| task_type | Multi-hop | GPTAcc | 300 | 0.3601 | 0.0000 |
| task_type | Single-hop Attribute Query | EM | 300 | 0.1255 | 0.0297 |
| task_type | Single-hop Attribute Query | Acc | 300 | 0.1322 | 0.0220 |
| task_type | Single-hop Attribute Query | F1 | 300 | 0.5356 | 0.0000 |
| task_type | Single-hop Attribute Query | Recall | 300 | 0.3775 | 0.0000 |
| task_type | Single-hop Attribute Query | Precision | 300 | 0.5437 | 0.0000 |
| task_type | Single-hop Attribute Query | GPTAcc | 300 | 0.4828 | 0.0000 |
| task_type | Subproblem Aggregation | EM | 300 | -0.0925 | 0.1098 |
| task_type | Subproblem Aggregation | Acc | 300 | -0.0962 | 0.0965 |
| task_type | Subproblem Aggregation | F1 | 300 | 0.2808 | 0.0000 |
| task_type | Subproblem Aggregation | Recall | 300 | 0.2187 | 0.0001 |
| task_type | Subproblem Aggregation | Precision | 300 | 0.2739 | 0.0000 |
| task_type | Subproblem Aggregation | GPTAcc | 300 | 0.2453 | 0.0000 |
| status | generation_error | EM | 325 | nan | nan |
| status | generation_error | Acc | 325 | nan | nan |
| status | generation_error | F1 | 325 | 0.0840 | 0.1308 |
| status | generation_error | Recall | 325 | 0.2403 | 0.0000 |
| status | generation_error | Precision | 325 | 0.0317 | 0.5688 |
| status | generation_error | GPTAcc | 325 | -0.0100 | 0.8571 |
| status | missing_final_answer | EM | 420 | nan | nan |
| status | missing_final_answer | Acc | 420 | 0.0871 | 0.0745 |
| status | missing_final_answer | F1 | 420 | 0.2886 | 0.0000 |
| status | missing_final_answer | Recall | 420 | 0.2490 | 0.0000 |
| status | missing_final_answer | Precision | 420 | 0.2875 | 0.0000 |
| status | missing_final_answer | GPTAcc | 420 | 0.1274 | 0.0089 |
| status | ok | EM | 755 | 0.0278 | 0.4457 |
| status | ok | Acc | 755 | -0.2832 | 0.0000 |
| status | ok | F1 | 755 | 0.2754 | 0.0000 |
| status | ok | Recall | 755 | 0.0100 | 0.7828 |
| status | ok | Precision | 755 | 0.3393 | 0.0000 |
| status | ok | GPTAcc | 755 | 0.0909 | 0.0125 |

## Suggested Next Checks

1. Compare these correlations within `OK`, `missing_final_answer`, and `generation_error` separately when interpreting process-quality vs answer-quality coupling.
2. If needed, add regression controls for `source` and `task_type` to test whether the `RTE` association remains after stratification.
