Entity Ambiguity Agreement Report
  Common overlapping query steps: 374
  Valid paired query steps: 374
  Excluded invalid pairs: 0

Binary Kappa (Yes/No)
  kappa: 0.822502
  observed_agreement: 0.919786
  expected_agreement: 0.548085
  human_counts: {"No": 244, "Yes": 130}
  llm_counts: {"No": 246, "Yes": 128}

Fine-grained Kappa (No + 4 ambiguity levels)
  kappa: 0.681386
  observed_agreement: 0.823529
  expected_agreement: 0.446131
  human_counts: {"No": 244, "Description": 18, "Object Identification": 58, "No Object Involved": 1, "Indirect Entity Ambiguity": 53}
  llm_counts: {"No": 246, "Description": 32, "Object Identification": 61, "No Object Involved": 1, "Indirect Entity Ambiguity": 34}