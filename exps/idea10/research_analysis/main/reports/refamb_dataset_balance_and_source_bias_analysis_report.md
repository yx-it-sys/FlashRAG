# RefAmb 数据构建与公平性验证的后续实验建议

## 背景结论
- 采用每个 `task_type=300` 的平衡采样用于比较“不同任务类型中的指代模糊”是合理的，不是理论必需但对公平比较很关键。
- `CRAG` 作为主要来源会带来来源偏置风险，需要额外分析证明结论不是由 source 分布驱动。

## 你应该补做的实验/分析

1. 双设置主结果对照
- 设置A：`task_type` 平衡集（你现在的 300/类）。
- 设置B：自然分布集（不强制平衡）。
- 目的：证明核心结论不依赖采样策略。

2. 按来源分层结果（必做）
- 汇报 overall + by-source（`crag / infoseek / mcsearch / oven`）。
- 对每个 `task_type` 的关键指标分别画图或列表。
- 目的：检查结论是否在各来源方向一致。

3. source-balanced 稳健性子集（强烈建议）
- 在每个 `task_type` 内尽量平衡 source（取交集或下采样）。
- 在该子集复现主结论。
- 目的：排除 CRAG 高占比导致的伪差异。

4. 统计控制分析（必做）
- 回归/方差分析：`metric ~ task_type + source + task_type:source`。
- 看 `task_type` 主效应在控制 `source` 后是否仍显著。
- 目的：给出“task_type 真效应”的统计证据。

5. 不确定性报告（必做）
- 对各指标报告 95% CI（bootstrap 推荐）。
- 对关键差异报告效应量（不仅 p 值）。
- 目的：说明 300/类结论稳定性。

6. 细粒度样本充足性检查（建议）
- 对要重点讨论的细分桶（如 `task_type × source × ambiguity_level`）统计最小样本数。
- 若过小，合并桶或只做定性分析。
- 目的：避免过度解读小样本现象。

## 论文中应明确写的限制与声明
- 平衡集用于“机制对比公平性”，不代表真实线上分布。
- CRAG 占比较高可能引入 domain/style bias。
- 已通过自然分布复现、source 分层、source-balanced 子集与控制分析验证稳健性。

## 建议交付物清单
1. 一张 `task_type × source` 计数热图。
2. 一张主结果对照图（平衡集 vs 自然分布）。
3. 一张 by-source 分层结果图。
4. 一个控制回归表（含 `task_type` 与 `source` 项）。
5. 附录中的 source-balanced 复现实验表。
