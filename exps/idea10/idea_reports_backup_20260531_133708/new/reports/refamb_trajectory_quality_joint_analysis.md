# RefAmb 轨迹质量联合分析报告

## 分析对象
- 热力图：`figs/refamb_trajectory_quality_original_main_heatmap.pdf`
- 雷达图：`figs/trajectory_quality_table_rte_lj_lines.pdf`

## 1. Disturb 分析（单独）

### 1.1 热力图视角（按 Original 正误分层）
0. 正确的样本RTE也会更高
1. 在 `Original-correct` 组，相比 Original，Disturb 和 Oracle 均出现退化：
- `Avg. RTE`: `0.565 -> 0.527`
- `Avg. LJ`: `1.000 -> 0.650`
RTE退化相较于Oracle要低，但LJ退化程度要更高
2. 除了S.A.Q和Comp.两个任务之外，Disturb会将更多的原本正确的样本变错。但S.A.Q.和Comp.比Oracle要稳一些

2. 在 `Original-incorrect` 组，Disturb 相比 Original 有稳定提升：
- `Avg. RTE`: `0.242 -> 0.319`
- `Avg. LJ`: `0.000 -> 0.136`
- `Flip ratio`: `0.136`（体现部分翻正能力）
说明 Disturb 在无 oracle 条件下具有独立纠错价值。

### 1.2 雷达图视角（按类别）
#### 1.2.1 轨迹质量
1. Disturb的轨迹质量在多数类别总体上要高于Original，在有些任务上甚至高于Oracle Intervention，体现适当的模糊能够激发Agentic Pipeline的探索能力。原因是适当模糊的查询召回的文档会更加多样化，有助于VLM的推理。
2. `S.A.Q.`和 `M.H.` 类别中 Disturb 的增益最明显，表明该类任务对轨迹扰动较敏感。
3. `Comp.` 类别增益较温和，提示复杂问题仍受检索覆盖或推理链质量约束。
#### 1.2.2 生成质量
1. 总体上Disturb干预后会使得回答质量降低，体现出高效、高活力的探索并未全部转化成最终生成质量的增益
2. 在`S.A.Q.`问题上，Disturb的探索转化成了生成质量的正收益
3. `S.A.`是子问题的聚合，模态耦合性较弱。趋近于纯文本问答问题，这类问题是大模型作为擅长的任务
### 1.3 Disturb 小结
Disturb 的定位是“轻量、普适、无 oracle 依赖”的纠错干预：有实际的轨迹活性收益，但也伴随对总体生成质量以及正确样本的潜在破坏。

## 2. Oracle 分析（单独）

### 2.1 热力图视角（按 Original 正误分层）
1. Oracle之后的LJScore和RTE值明显低于Original，包括Flip Ratio也总体上低于Disturb
说明Ground Truth Entity的相关信息并不绝对地对VLM来说是一个正向的信号，正确，不代表有用。
2. 自身回答正确的样本，哪怕指代是模糊的，也能正确回答，Oracle干预甚至会减弱：VLM有自己的解决问题的方式和参数化知识，干预与 VLM 自身策略产生冲突

3. 在 `Original-incorrect` 组，Oracle 提供强于Disturb的提升能力：
- `Avg. RTE`: `0.242 -> 0.355`
- `Avg. LJ`: `0.000 -> 0.238`
- `Flip ratio`: `0.238`（高于 Disturb 的 `0.136`）
说明 Oracle 在翻正与质量提升上更强。

### 2.2 雷达图视角（按类别）
1. 轨迹质量与生成质量方面，Oracle均呈现出稳定优于 Original 的表现，说明更加精确的实体会带来更加稳定的收益和轨迹活性
2. `S.A.` 类别对 Oracle 响应最强，RTE 与 LJ 都达到最高梯度。
3. `Comp.` 类别虽有改善但幅度有限，对比类问题两种实体的信息要均等才能成功回答问题，单给出图片中的实体是不够的

### 2.3 Oracle 小结
Oracle 的定位是“信息更充分、上限更高”的强干预：在纠错与稳定性上整体优于 Disturb，但仍需避免对正确样本的过度改写。

## 3. Disturb 与 Oracle（仅作简短对照）
1. Disturb：无 oracle 前提下可获得可迁移收益。
2. Oracle：在相同框架下进一步抬升上限并改善稳定性。
3. 二者应作为两条独立结果线汇报，再在结论处做强弱关系对照。

## 4. 可直接用于论文正文的简述（中文）
在 RefAmb 上，Disturb 与 Oracle 均应作为独立干预进行报告。Disturb 在 `Original-incorrect` 子集上已带来可观纠错收益（RTE/LJ 与翻正率提升），证明其无 oracle 条件下的有效性；Oracle 在相同评估切片上进一步提升多数类别与总体指标，并在正确样本子集上表现出更好的稳定性。两者共同表明，轨迹干预既存在可部署的基础增益，也存在可达的上界增益。

## 5. 风险与说明
1. 雷达图是类别聚合结果，热力图是按初始正确性分层结果；两者不可逐格对应。
2. 建议在正文显式定义 `Flip ratio` 的方向语义（翻错/翻正）与分母口径。
