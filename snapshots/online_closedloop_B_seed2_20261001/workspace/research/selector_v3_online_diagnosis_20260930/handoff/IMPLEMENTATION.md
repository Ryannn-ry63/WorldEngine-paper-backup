# 覆盖、奖励执行与更新时序诊断实现

新目录 `research/selector_v3_online_diagnosis_20260930` 从已运行的 reward-v2 复制源代码，并记录原始 657 个源文件哈希。原 v1/v2 的代码、运行产物、训练 checkpoint 和评测不修改。模型、地图、场景及渲染资产复用已登记的只读来源，没有复制大型资产。

本轮目标是定位在线下降的原因并完成受控实验，不能在看到新评测前保证分数提升。

## 已知证据和待验证假设

原调度为 `(rank + round) % N`：8 卡相邻轮会重叠 7 个场景，64 episodes 不能保证 25 个训练场景的均匀覆盖。新调度按 seed 固定打乱一次，以 `round*8+rank` 的全局游标循环遍历；任何前缀的各场景访问次数相差不超过 1。恢复沿用同一游标。

已有 v2 的 64 episodes 小开发集结果下降，但仅凭单个 DAC 失败场景不能证明奖励公式错误，也不能证明频繁重规划就是根因。在线采样会改变训练分布，更新也会改变后续观测，学习方法相同不意味着分数必然不降。

本版本没有再重写奖励：20 条候选在同一个实际根状态独立运行完整 4 秒响应式交通，保留 v2 专家进度参考和 PDM 组件。主环境只提交所选候选的第一步，然后重新观测/决策。候选 4 秒固定执行分数衡量该候选执行后果，**不等同于接下来不断重规划的策略回报**；诊断将测量这项差距。

## 实验设计与代码落点

| 项目 | 实现 | 当前验收边界 |
|---|---|---|
| 233 个静态候选 | `handoff/prepare_scene_pool.py` | train 150 / 25 logs；dev 83 / 12 logs；5 个 holdout logs 隔离 |
| 真实技术准入 | `innovation3/coverage_probe.py`、`fixed_worker.py` | H100 上逐场景渲染、full20、8 decisions；未运行不记作通过 |
| 冻结覆盖合同 | `innovation3/coverage_manifest.py` | 同一 log 只允许在一个 split；同 split 可多个 clip |
| A/B 更新时序 | `innovation3/fixed_learning.py` | A 每步更新；B 固定策略采集 8 步，缓存上下文后顺序重新前向、更新 8 组 |
| 更新审计 | `live_learning.py`、`provisional_session.py` | 分别核验观测动作版本、优化次数和反馈；无信号组不虚增版本 |
| 覆盖与恢复 | `ddp_session_boundary.py` | 3 / 6 遍全 rank 提交标记；恢复可补发未完成写入的标记 |
| 相同执行轨迹双评分 | `worldengine/online/scorer_parity.py` | 调用实际 `MetricManager.compute_pdm_scores`；对齐状态后容差 1e-6 |
| 根状态诊断 | `coverage_probe.py` | 12 dev logs 各选 2 clip，每 clip 2 roots；全20奖励/双模型 logits/候选 oracle |
| 首步后重规划 | `coverage_probe.py`、`fixed_worker.py` | 从 reset 重放到同根并验证候选/状态哈希，再执行 parent / online64 / oracle 首步，统一用冻结 parent 继续 |
| 计划切换诊断 | `plan_diagnostics.py`、`navformer_client.py` | 比较绝对坐标共同未来时刻；1 秒诊断复用原计划后缀，观察/推进仍 0.5 秒 |
| 多节点运行 | `handoff/run_job.py`、`run_queue.py` | 每任务独立锁、临时目录、输出、进程组、log；不同节点领取不同任务 |
| 完整评测 | `evaluation_artifacts.py`、`summarize_experiment.py` | 全 dev R/NR；三种子固定门槛；正式四表精确覆盖 |

B 组保留 V3 的 exact-group loss、AdamW、学习率 `3e-5`、KL `1e-3`、梯度裁剪和冻结参考。缓存中只有原 selector 所需的当前/历史输入；奖励在反馈后另行提供。每个缓存组只学习一次，不加 PPO ratio、value、GAE 或切换惩罚。模型采集时的概率与训练时当前 logits 分别记录，不拿旧的计算图做更新。

## 诊断解释限制

- 48 roots 从预先固定的开发日志抽样；追加掉分/安全退化场景单独记录，不混入代表性 48 roots 的统计。
- 候选 oracle 只是给定 20 条候选中的最高评分上界，不作为训练或正式评测策略。
- 首步后续诊断采用**相同冻结 parent continuation**，是受控动作对照，不声称是每个首步模型自身的完整策略价值。
- 诊断复用原训练桥接的 map-only warmup；它与 legacy evaluator 的初始历史方式不同。正式效果仍通过原 evaluator 测量，诊断回放不冒充正式评测失败现场。
- legacy evaluator 将加速度/部分动态量填零。双评分既记录原 legacy 数值，又提供动态量对齐路径；对齐后所有组件须一致，原值差异须由 comfort 解释。未改正式 scorer。
- 1 秒计划复用只属于诊断作业；所有主训练和正式四表仍用 0.5 秒更换计划。
- development 包含历史训练来源日志，不能包装成独立最终测试。原 holdout 与正式测试不参与训练或方法调参。

## 门槛与最终结果

每个方法跑 seed 0/1/2。开发集三种子平均相对父模型要求 R 分数至少 +0.005、NR 不低于 -0.005，且 R/NR 的 NC、DAC、TTC 均不低于 -0.005。两法都满足时按 R 选择，精确相同时选 A。选择后冻结方法并评测全部三个种子；不能看测试结果再挑种子。

预算为全实验 18 小时墙钟时间；150 场景全通过时分别在 456、904 episodes 检查。若最终 6 遍未完成，只报告同预算 3 遍对照，不将其自动升级为正式最终选择。

当前实际验证结果见 `registry/implementation_status.json`。运行入口与恢复命令见 [H100_COMMANDS.md](H100_COMMANDS.md)。当前没有新 H100 训练、开发集或正式四表结果，因此没有新性能提升结论。
