# Baseline 实现逻辑说明（RefAmb / OmniSearchPipeline）

本文档对应当前代码实现，覆盖以下三个入口：

- `scripts/run_baseline_all_sources.sh`
- `run_refamb_by_source.py`
- `flashrag/pipeline/omni_pipeline.py`（`OmniSearchPipeline`）

---

## 1. 运行编排层：`run_baseline_all_sources.sh`

该脚本负责按 source 串行执行 baseline，并统一收集日志。

### 1.1 核心职责

1. 解析运行参数（均支持环境变量覆盖）：
   - `REFAMB_MODEL`（默认 `intervl3_5_8b`）
   - `REFAMB_RUNNER`（默认 `run_refamb_by_source.py`）
   - `REFAMB_TRAIN_PATH`
   - `REFAMB_SOURCES`（默认 `crag infoseek mcsearch oven`）
   - `REFAMB_CONTINUE_ON_ERROR`（默认 `1`，单源失败后继续）
2. 为每个 source 启动一次 Python runner。
3. 将每个 source 的 stdout/stderr 分别保存到 `${LOG_DIR}/${source}.log`。
4. 每个 source 结束后 sleep 一段时间并打印 `nvidia-smi`，帮助释放/观察显存。

### 1.2 鲁棒性设计

- 脚本实现了 `safe_exit`：即使被 `source` 执行，也不会直接 kill 掉交互终端。
- 当 `CONTINUE_ON_ERROR=1` 时，某个 source 出错不影响后续 source 执行。

---

## 2. 单源执行层：`run_refamb_by_source.py`

该脚本负责“一个模型组 + 一个 source”的完整实验。

### 2.1 配置选择与模型路由

脚本通过三组映射控制实验：

1. `MODEL_TO_SOURCE_CONFIG`：模型组到四个 source 配置文件路径的映射  
   （已包含 `qwen2_5_7b / qwen3_5_scaling / llava15_7b / intervl3_5_8b`）。
2. `MODEL_TO_SOURCE_SAVE_NOTE`：每组模型对应的结果命名后缀。
3. `MODEL_TO_SOURCE_RESUME_DIRS`：可选历史结果目录，用于跨目录断点续跑过滤。

### 2.2 数据过滤与断点续跑

给定 `--source` 后，执行如下过滤：

1. 从 `train-path` 读取 jsonl，仅保留 `obj["source"] == source` 的样本。
2. 若未禁用 resume，则读取历史 `omnisearch_trajectories.jsonl`，收集已跑 `id`。
3. 若未禁用 incremental，则读取当前输出目录 `omnisearch_trajectories.jsonl`，收集已跑 `id`。
4. 用并集 `skip_ids = resume_ids ∪ existing_ids` 过滤，避免重复推理。

### 2.3 运行前环境设置

1. 清理代理环境变量（避免外部代理影响本地推理）。
2. 从目标 config 读取 `gpu_id` 并预设 `CUDA_VISIBLE_DEVICES`。
3. 构建 `Config` 时覆盖关键字段：
   - `source`
   - `split = ["task_balanced_analysis_subset"]`
   - `save_note`
   - `data_dir / save_dir`
   - `roi_preprocess_config.enabled = false`（baseline 默认关闭 ROI 预处理）

### 2.4 Pipeline 装配与安全检查

1. 通过 `get_generator(config)` 和 `get_retriever(config)` 初始化组件。
2. 构建 `OmniSearchPipeline(config, retriever, generator)`。
3. 强制检查 pipeline 类名：必须是 `flashrag.pipeline.omni_pipeline.OmniSearchPipeline`，防止误用 EAO/VORS 变体。
4. 调用 `pipeline.run(test_data, do_eval=True)` 执行完整推理与评估。

---

## 3. 核心推理层：`OmniSearchPipeline`

该模块实现 ReAct 风格的“生成-检索-再生成”闭环。

## 3.1 初始化阶段

主要完成以下工作：

1. 读取 prompt 模板（`omni_prompt_path + omni_prompt_version`）。
2. 设置输出路径并创建：
   - 轨迹文件：`omnisearch_trajectories.jsonl`
3. 设置文本长度控制参数：
   - `omni_retrieval_char_limit`
   - `omni_retrieved_doc_char_limit`
4. 读取 source 过滤目标（`config.source`），并做名称规范化：
   - `mc-search -> mcsearch`
   - `info-seek -> infoseek`
5. ROI worker 相关状态初始化（baseline 默认关闭）。

## 3.2 数据级 run 流程

`run(dataset, do_eval=True)` 执行顺序：

1. 先按 `source` 做样本过滤（特别地，`infoseek` stage 允许匹配 `oven`）。
2. 对每条样本调用 `iterative_infer(question, id, image_id)`。
3. 收集 `pred` 后调用 `evaluate(...)`。
4. 输出总耗时、均耗时，并记录到 `records.txt`。

## 3.3 单样本 iterative_infer 流程

### Step A：输入构建

1. 从 `${data_dir}/${dataset_name}/images/{image_id}.jpg` 读图。
2. 组装初始消息：
   - system：任务指令 prompt
   - user：`Input Question: ...` + image
3. 若图片缺失，直接写入 `status=missing_image` 的轨迹记录。

### Step B：首轮生成

1. 调用 `_generate_text(messages)`：
   - 对可重试 API 错误做指数退避重试（最多 5 次）。
2. 对输出执行 `_truncate_after_search`，防止 `<Search>` 后粘连冗余段落。
3. 记录 `"First Response"`，解析并写入动作节点（`thought/sub-question/search/final_answer`）。

### Step C：最多 5 轮 ReAct 循环

循环条件：未出现 `Final Answer` 且 `turn < 5`。

每轮执行：

1. 解析 `<Search>` 动作，得到：
   - `text_retrieval`
   - `image_retrieval`
   - `no_retrieval`
2. 若为文本检索：
   - `retriever.search(query_type=text, topk=text_retrieval_topk)`
3. 若为图像检索：
   - （可选）ROI 预处理
   - `retriever.search(query_type=image, topk=image_retrieval_topk)`
4. 检索结果格式化：
   - 文档字段优先级提取
   - 每文档截断 + 全局截断
   - CRAG 图像结构化结果走实体属性专用格式化
5. 写入 trajectory 的 retrieval_result 节点。
6. 构造 follow-up user message（携带检索证据或 no-retrieval 约束）。
7. 再次生成 -> 截断 -> 记录 `"Response"` -> 解析动作节点。

### Step D：终止与状态落盘

1. 若解析到 `Final Answer`：
   - `status=ok`
   - 保存 `final_answer` 与完整 trajectory
2. 若到达最大轮次仍无 final：
   - `status=missing_final_answer`
   - 保存最后响应与 trajectory
3. 若生成或调用异常：
   - `status=api_error` 或 `generation_error`
   - trajectory 追加 `error` 节点并落盘

---

## 4. 轨迹记录格式（Baseline 可复现关键）

每条样本写一行 JSONL，核心字段：

- `question`
- `id`
- `final_answer`
- `status`（`ok / missing_final_answer / generation_error / api_error / missing_image`）
- `duration_seconds`
- `trajectory`（动作序列）

`trajectory` 中常见 `action`：

- `thought`
- `sub-question`
- `search`
- `text_retrieval_result`
- `image_retrieval_result`
- `no_retrieval_result`
- `final_answer`
- `error`

---

## 5. 与论文 Setup 对齐时可直接表述的实现要点

可用于论文的简洁描述：

1. Baseline 使用统一 ReAct 控制循环（最多 5 轮），轮内执行“动作生成 -> 动作解析 -> 条件检索 -> 证据回注 -> 再生成”。
2. 所有中间行为与检索证据均写入结构化 trajectory（JSONL），并带状态码，支持失败分析与断点续跑。
3. 训练/测试分源执行，通过 source 级过滤和历史 ID 过滤保证跨次运行可增量复现。
4. 默认关闭 ROI 预处理，确保 baseline 不引入 VORS/EAO 类附加控制模块。

