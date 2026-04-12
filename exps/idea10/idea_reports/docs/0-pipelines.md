0. 复现 Omnisearch Pipeline
    - 脚本：run.py
1. 人工标注120条数据，标注查询是否实体模糊与实体模糊类型
2. 使用LLM进行全量标注
    - 脚本：/home/you/FlashRAG/exps/idea10/scripts/label_entity_ambiguity_with_llm.py
    - 输入：trajectory_annotation.jsonl
    - 输出：trajectory_annotation.llm_labeled.jsonl
3. 计算 κ
    - 脚本：/home/you/FlashRAG/exps/idea10/scripts/compute_entity_ambiguity_kappa.py
    - 输入：（fixed）trajectory_annotation_manual_sample.jsonl，（fixed）annotation_results_readable.jsonl，trajectory_annotation.llm_labeled.jsonl
    - 输出：stats on shell
4. 分析标注结果
    - 脚本：/home/you/FlashRAG/exps/idea10/scripts/analyze_entity_ambiguity_category_metrics.py
    - 输入：intermediate_data.json，omnisearch_trajectories.jsonl，label/trajectory_annotation.llm_labeled.jsonl
    - 输出：/home/you/FlashRAG/exps/idea10/idea_reports/下的图和报告
5. LLM重写查询
    - 脚本：/home/you/FlashRAG/exps/idea10/scripts/rewrite_entity_ambiguous_queries_with_llm.py
    - 输入：trajectory_annotation.llm_labeled.jsonl
    - 输出：trajectory_annotation.llm_labeled.rewrite_deepseek.jsonl
6. Oracle实验：将第一轮实体模糊检索用rewrite Query替代，后面让Agent自行迭代
    - 脚本：/home/you/FlashRAG/exps/idea10/scripts/exp_oracle_v1/run_first_round_oracle_rewrite_experiment.py
    - 输入：/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment,label/trajectory_annotation.llm_labeled.rewrite.jsonl
    - 输出：
