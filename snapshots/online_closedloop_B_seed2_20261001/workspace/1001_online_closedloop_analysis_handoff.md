# 在线闭环 A/B 实验：结果分析交接（2026-10-01）

本文供新会话接手结果分析。当前任务是理解已经完成的实验、核对评测完整性、定位系统性问题，再决定下一轮；不要一上来重新训练或改奖励。

## 1. 当前进度：训练完成，评测登记仍有两个缺口

用户反馈：A/B 各自 s0–s2，包括评价都已跑完。交接时于 **2026-10-01 08:29 UTC** 只读核查共享 registry，发现以下状态。新会话应先核对差异，不能直接假设用户漏跑，也不能把平台 exit code 0 当成全表完成。

| 项目 | 交接时磁盘证据 |
|---|---|
| A_s0、A_s1、A_s2 | 全部 COMPLETE；每组 904 episodes |
| B_s0、B_s1、B_s2 | 全部 COMPLETE；每组 904 episodes；B_s2 已恢复完成 |
| 训练覆盖 | 六组均访问 150 个场景；每场景至少 6 次、最多 7 次 |
| p3 / p6 checkpoint 标记 | 六组均有；分别在 456 / 904 episodes 提交 |
| A/B 开发评测 | 24 项中 22 项 COMPLETE |
| 当前未登记的两项 | `dev_train_B_s2_p6_R`、`dev_train_B_s2_p6_NR`：未找到对应 status.json |
| 基线与决策间隔诊断评测 | 8 项全部 COMPLETE |
| roots48 | 修复后的作业 COMPLETE，48 个根、144 条 continuation |
| roots_extra_0…3 | 无完成登记；不能当作已经分析过额外失败场景 |
| 正式四表 | 注册的 16 项均无完成登记：parent 和选定方法三个种子，各四表 |
| 方法选择 | `selection.json` 尚不存在；summary 为 `PENDING_FULL_P6_THREE_SEED_DEV` |

总计 63 个注册作业，41 个 COMPLETE，22 个无状态文件。后者包括 2 个开发评测、4 个额外诊断、16 个正式评测，不能统称为“22 个运行失败”。

本次已重新核验 **30 项完成的开发/间隔评测**：status 指向的 result 文件哈希、所引用 CSV 哈希、`PASS_EXACT_COVERAGE`、result 中的 83 个预期场景集合、数值有限性、逐场景均值一致性，全部通过。尚未重新解析所有原始 CSV 的语义，也未逐一重算所有训练 checkpoint/逐轮回执；这些不属于本次文档交接的验证范围。

配套文件：

- [只读登记快照与结果路径索引](research/online_analysis_handoff_20261001/registry_snapshot.json)：各 job 状态路径/哈希，训练进度/覆盖，已验收 result/CSV 路径与哈希，模型来源，以及交接时 summary 副本。
- [30 项评测均值 CSV](research/online_analysis_handoff_20261001/development_means.csv)：便于新会话立即开始作图、对照。

**首先排查 B_s2 p6 是否在其他 attempt/输出位置完成但未登记。** 现有恢复队列空闲 120 秒会退出；如果当时 B_s2 p6 尚未就绪，其他评测节点可能先退出。这是有代码依据的可能解释，尚未逐个核对节点退出日志，不能写成已确认根因。只在确认没有可验收结果后补跑缺失两项，不重训六组模型。

## 2. 论文主线与用户已经明确的目标

主线是 **V2 → 离线 V3 选择器 → 在线闭环**。保留 V3 的候选生成器、选择器和学习方法，在实际交互到达的状态上，让候选进入响应式 simulator，按照稳定评分规则评价执行后果，再在线更新，最终用原评测检验收益。

学长建议的重点是“快速把候选轨迹部署进 simulator，拿到更真实的 reward”。引入 simulator 的价值在于模拟候选执行之后自车和周围交通的未来状态；不是为了任意设计一套新奖励，也不是把一个短时代理分数改名为精确奖励。

专家未来可以作为离线可获得的评分参考，只要不进入策略输入、候选生成、控制器或替代策略执行，就不妨碍在线闭环训练。用户已认可这个边界，不要重新纠结“用了专家 EP 就不是在线”。

用户关心完整场景覆盖、跨种子说服力、效率和最终论文证据。可以安排多个独立 8×H100 节点并行，但应先利用已有结果形成明确问题，再安排有区分力的实验。不要只凭一个失败场景加惩罚项；也不要因为方法继承自 V3，就承诺在线训练分数必然不下降。

## 3. 路径与阅读顺序

所有相对路径均相对于：

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
```

| 简称 | 相对路径 | 用途 |
|---|---|---|
| EXP | `research/selector_v3_online_diagnosis_20260930` | 当前 A/B 冻结实验与主要 registry |
| REG | `research/selector_v3_online_diagnosis_20260930/registry` | 作业、场景合同、结果、checkpoint、汇总 |
| REPAIR | `research/selector_v3_online_diagnosis_repair_20260930` | simulator 根恢复、数值审计和存储恢复补丁 |
| V2 | `research/selector_v3_online_simreward_v2_20260930` | 专家参考 reward-v2，历史 64ep 模型 |
| V1 | `research/selector_v3_online_simreward_20260929` | 更早的参考车 reward 方案，仅用于历史对照 |
| MotionRFT | `../MotionRFT` | 学长提供的源码；不能据此宣称当前已换成其原生后端 |

建议阅读顺序：

1. 本文、配套快照；`REG/summary.json`、`REG/diagnostic_summary.json`。
2. `EXP/handoff/IMPLEMENTATION.md`：受控实验定义；末尾“尚未运行”是旧状态，已被实际结果覆盖。
3. `REG/experiment.json`、`REG/scene_manifest.json`、`REG/runtime.json`：实验、场景与配置合同。
4. `EXP/handoff/summarize_experiment.py`、`evaluation_artifacts.py`：验收与配对统计、选模规则。
5. `REPAIR/storage_recovery_20261001/RECOVERY_COMMANDS.md`：当前恢复入口和队列语义；其中 B_s2=824/904 是恢复前旧描述，现在已到 904。
6. `REPAIR/REPAIR_STATUS.md` 和各 hotfix manifest：工程变更来源；同样注意旧文档状态不能覆盖最新完成证据。

`REG/experiment.json` 中 `REGISTERED_NOT_RUN` 是初始冻结元数据，不能拿来判断今天是否跑过。实际接受哪个 attempt，以 `REG/jobs/<job>/status.json` 指向的 result 为准，不以 glob 最新目录或日志最后一行代替。

当前实验 SHA256：`bc821c9f89a77b88536ab291d6ec963a0133ecb82f0ab1fefdb69ace966fde41`。

新的分析脚本、图表和报告放到新的分析目录。不要直接修改 EXP 的冻结代码/handoff、结果 CSV、已有选择文件或哈希合同。

## 4. 当前实际训练与奖励，避免混淆版本

### 4.1 一次决策发生什么

1. 从在线主环境当前观测生成 20 条候选。
2. 每条候选恢复同一个实际根状态，在响应式交通中执行完整 4 秒，即 8×0.5 秒；记录根帧及未来帧共 9 帧。
3. 根据各分支实际自车状态、周围车辆状态和地图，计算 PDM 组件及最终奖励。候选评分分支不逐步渲染、不逐步调用策略，因此比完整视觉闭环重跑 20 遍便宜。
4. 选择器选择动作；主环境只提交所选轨迹前 0.5 秒，随后重新观测和决策。每个训练 episode 有 8 次决策。
5. 按 A/B 定义进行学习更新。

奖励合同：`fixed_candidate_4s_logplay_pdm_expert_v2`。主代码在：

```text
EXP/code/projects/SimEngine/worldengine/online/expert_segment_reward.py
EXP/code/projects/SimEngine/worldengine/online/expert_fixed_rollout.py
EXP/code/projects/SimEngine/worldengine/online/fixed_parallel.py
EXP/code/projects/SimEngine/worldengine/online/scorer_parity.py
EXP/code/projects/SimEngine/worldengine/online/plan_diagnostics.py
```

实际调用仓库 `PDMScorer`。合同描述公式为 `G*(5*G*EP + 5*TTC + 2*C)/12`，`G=NC*DAC`；分析应同时读取 scorer 返回的 gated_EP 和 raw EP，不在报告外另造一个公式重算。EP 用评分专用的日志专家窗口作为参考，进度按候选与参考进度的较大值归一化；小于阈值时沿用 scorer 约定。专家窗口不进入策略。

NC/DAC 检查整个实际分支（含端点）；TTC 沿用仓库的投影检查，并使用分支中的实际交通状态；comfort 使用实际速度与加速度。当前没有额外跑一辆 IDM 进度参考车，也没有“参考车失败后取最后安全进度”。

这里的“精确”指 **指定模拟器、控制器、离散步长和评分规则下，完整 4 秒候选执行的计算值**。它不等于现实世界真值，也不等于未来持续重规划策略的价值。当前实现是 WorldEngine 响应式/headless 分支执行，借鉴 MotionRFT 快速评分思想；不能写成已原样迁入 MotionRFT 的 JAX/numax simulator 或完整原版奖励规则。

原 evaluator 的加速度等部分动态量存在填零行为，训练保留实际动态量。因此“评分代码对齐”仍有明确输入差异；诊断同时保存 legacy 原值和动态量对齐后的评分，不能修改正式 scorer 来消除差异。

### 4.2 A/B 到底比较什么

| 项目 | A | B |
|---|---|---|
| 更新时机 | 每次决策后更新 | 冻结策略采集完整 8 步，再按时间顺序重新前向、逐组更新 |
| episode 内后续采样 | 受前面更新后的策略影响 | 由采集时冻结策略决定 |
| 实际重规划间隔 | 0.5 秒 | 0.5 秒 |
| 奖励、候选、损失 | 继承 reward-v2 / V3 | 同左 |

B 不是一次执行整条 4 秒轨迹，也不是将决策间隔改为 1 秒；它只改变更新时序。缓存当前/历史 selector 输入，训练时重新计算当前 logits，不复用旧计算图。保留 exact-group loss、AdamW、学习率 `3e-5`、KL 系数 `1e-3`、冻结参考与冻结生成器；没有加入 PPO ratio、value、GAE 或轨迹切换惩罚。

六组均从 **同一个离线 rare_tuned V3 seed0 父模型** 出发，s0/s1/s2 是本轮在线实验种子，不是三个独立离线预训练父模型。父 selector：

```text
experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/train/epoch_64_scene_selector.pt
SHA256: aed5e36f1ccb0df96b703a901b54f48c41dcf1967cc86d63519eb4da37519eb2
```

诊断中的 `online64` 指 V2 下 `reward_v2_seed0_20260930T053704Z_3664064.eval64/online64.pth`，不是 V1 64ep，也不是本次 A/B 的中间模型。

### 4.3 覆盖预算

训练 150 场景 / 25 个 origin logs；开发 83 场景 / 12 logs，另隔离 5 个 holdout logs。开发集含历史训练来源日志，只供开发诊断/选模，不能包装成独立最终测试。

调度采用 `seeded_cyclic_global_episode_v2`：按 seed 固定打乱，以 `round*8+rank` 全局游标遍历。预算为 `8*ceil(passes*150/8)`，所以 p3=456、p6=904，而不是之前讨论的 64→256。现在六组已完整覆盖 6 遍，没有理由仅因旧计划提到 256 就再启动一次。

checkpoint 定位用 `REG/jobs/train_<A|B>_s<0|1|2>/training.coverage3.json` / `training.coverage6.json`；进度用 `training.progress.json`；实际逐轮报告和状态在 `training.attempts/`。不能从 episodes 数直接推断有效 optimizer steps：无信号组、恢复、各 rank 回执要单独核对。

## 5. 已遇到的问题及处理边界

| 问题 | 已处理内容 / 分析时注意 |
|---|---|
| 启动导入 worldengine 失败、后台不易看到错误 | 已修复路径和预检；新入口前台输出并落日志，不复用早期 nohup 命令 |
| 老调度覆盖严重重叠 | 原 `(rank+round)%N` 相邻轮重叠 7/8；当前已改全局游标并扩至 150 场景 |
| V1 参考车失败 33/512 组 | 额外 IDM 进度参考及最后安全截断可能偏置；V2 已改专家参考，不能把此问题当成本轮仍在发生 |
| roots 重放状态哈希不一致 | 根因涉及新 simulator 静态对象身份、episode_start 等；修复为复用同场景初始 snapshot/static bundle；没有删哈希字段绕过验收 |
| CORRECTNESS_STOPPED_AT_BOUNDARY 被平台记成功 | 当时不等于训练完成；已修复状态/退出行为，修复后的 H100 roots48 完成才释放训练 |
| H100 logits 数值审计过严 | A_s2/B_s1 及后续 B_s2 使用登记 hotfix：atol 从 1e-5 到 5e-5，rtol=1e-4；不改学习公式，分析保留补丁/恢复来源 |
| Disk quota exceeded | B_s2 与评测曾因写文件失败；清理后恢复。现在 B_s2 已 904，无须再按旧 824 断点续训 |
| 多评测节点重复领取任务 | 恢复入口统一原子 mkdir 互斥；旧 flock 的跨节点表现不可依赖；不能盲删远端锁 |
| 截止时间过期 | 原 18 小时 campaign 之后有登记恢复窗口；报告应披露中断和恢复，不宣称全部严格在原 18 小时内完成 |
| 本地图像未长期保留 | 恢复评测将临时图像放本地 scratch；数值、计划、配置、日志验收归档。metadata 中的临时图像路径失效不必然代表结果损坏 |

补丁证据集中在 REPAIR 的 `evidence/`、`hotfix_logit_tolerance_20261001/`、`hotfix_logit_tolerance_bs2_20261001/`、`storage_recovery_20261001/`。这些属于工程故障与恢复，不应与“奖励目标是否适合策略闭环”的科研问题混为一谈。

## 6. 已有初步信号：总分提升与安全退化并存

以下来自交接时 registry 汇总；是 83 场景开发集相对 parent 的三种子平均变化，**0–1 原始尺度**。例如 +0.03349 相当于 100 分尺度 +3.349 分。不是正式四表结论，也不是因果归因。

| 预算/方法 | ΔR PDMS | ΔNR PDMS | ΔR DAC | ΔNR DAC | ΔR EP | ΔNR EP | 当前门槛 |
|---|---:|---:|---:|---:|---:|---:|---|
| p3 A | +0.02484 | +0.02233 | -0.04418 | -0.04418 | +0.11283 | +0.11163 | 不通过 |
| p3 B | +0.01644 | +0.02434 | -0.04418 | -0.02811 | +0.09248 | +0.09959 | 不通过 |
| p6 A | +0.03349 | +0.01793 | -0.03614 | -0.04819 | +0.13600 | +0.12396 | 不通过 |
| p6 B | 待 s2 的 R/NR 登记 | — | — | — | — | — | 尚不能判定 |

p6 A 的 NC 变化约 R -0.01004 / NR -0.00602，TTC 约 R -0.01205 / NR -0.02008；comfort 均值变化为 0。不能只展示 PDMS 的正增益而忽略安全组件。

目前更适合的问题是：**进度收益为何伴随可行驶区域/部分安全指标退化，这种取舍是否跨场景、跨种子稳定；改变更新时序能缓解多少？** 不是预先判定“全部训练无效”，也不是看到平均分提高就结束诊断。

旧 online64 在这 83 场景上相对 parent 的 PDMS 为 R +0.00917 / NR +0.02602，DAC 均为 -0.03614。此前小开发集报告过最差模式 Δ=-0.064285；两个结论的样本集不同，不能混在同一张同预算结果表中。

### 6.1 roots48 已提供的证据

接受的 attempt：`REG/jobs/roots48/attempt_20260930T153305Z_158d9f37`。

- 24 clips / 12 logs，每 clip 根决策 0 和 4，共 48 roots。
- 全部候选奖励为零：3/48；parent 与 online64 选择不同：14/48。
- 平均候选 oracle gap：parent 0.13535，online64 0.13898。
- 动态量对齐后的 scorer parity 最大误差：0。
- 共同未来时刻的计划几何平均差约 1.570 m；下一步差约 0.250 m。这是描述量，本身没有证明切换导致 DAC。
- 每根分别强制 parent/online64/oracle 首步，再统一冻结 parent continuation，共 144 条；`replanned_reward - fixed_reward` 平均分别约 -0.03549 / -0.02772 / -0.19876。

oracle 选中的完整固定候选，首步后接相同 parent continuation 时损失更大，提示“固定候选价值”与“首步+重规划后果”可能不一致。但这是 old online64/parent 的受控诊断，**不是本次 A/B p6 的直接证据**；continuation 也不是各模型自己的后续策略。根来自 headless map-only warmup，不能冒充正式 evaluator 的真实失败现场。

### 6.2 1 秒间隔诊断

相对各自 0.5 秒基线，1 秒计划复用的 PDMS 增量：parent R +0.09189 / NR +0.10901，online64 R +0.08079 / NR +0.08205。两者都改善，说明值得研究执行/重规划接口，但不能只归因于在线学习或直接把主实验间隔改成 1 秒。需比较交互效应、DAC/NC/TTC 和轨迹连续性，同时承认复用计划还改变了候选生成/执行过程。

## 7. 新会话的分析顺序

### 第一步：收齐证据，确认所比较的是同一套实验

核对上述缺失两项及其他输出位置；完成态必须有可验收 result、正确模型哈希、正确场景集合和组件数值。B_s2 的 p6 应确实来自 coverage6，而非 p3 或旧模型导出。没有现成结果才补这两项。

核对训练 marker → checkpoint → selector 导出 → eval 配置 → result 的链路；比较各 seed 的场景清单、奖励合同、执行间隔、学习配置和 hotfix。逐轮分析只读 committed progress 所指向的有效回执，排除失败 attempt 的重复数据。

`summarize_experiment.py` **会写 summary、diagnostic_summary、REPORT，并可能不可变地生成 selection**；不是纯读操作。先做独立只读分析，再决定是否通过现有入口更新正式登记。

### 第二步：完整报告总体效果和异质性

建立 parent、old online64、A/B×s0/s1/s2×p3/p6×R/NR 表格；给出绝对值、配对 delta、跨种子均值/标准差。缺失结果显式留空，不用两种子平均替代三种子。

以 origin log 为聚类单位做配对 bootstrap，避免同一日志的多个 clip 当作独立证据。原 summary 已有固定 seed 的 5000 次 log-cluster bootstrap；复用其定义并说明它与训练种子方差是不同不确定性来源。开发集只有 12 logs，应报告区间与局限。

逐场景比较 NC/DAC 的成功→失败、失败→成功，TTC/EP/总分变化；统计跨 seed 重复失败、R/NR 共性和差异。用逐场景联合安全条件分析进度，不用平均 NC×平均 DAC 代替联合通过率。对安全子集的条件统计需固定子集定义，防止各模型不同子集造成选择偏差。

重点看 p3→p6 是否稳定改善、是否扩大 EP 与安全的冲突；确认少数 origin logs 是否支配均值。不要只挑最好种子，也不要用单个最差场景代表整个方法。

### 第三步：分清“奖励计算正确”与“奖励优化目标合适”

沿代码/回执核对坐标系、车辆盒、地图范围、时间轴、根是否已违规、候选恢复一致性、EP 参考重锚定、所有零奖励组、首次 NC/DAC/TTC 违规时刻。若发现实现错误，先量化影响范围。

若计算一致，进一步检验：

1. 给定 20 候选是否存在安全且高分选项？没有的话，selector 单独学习未必能解决；不要预先将问题归因于排序。
2. 有好候选时是否选错，oracle gap/排序相关性是否改善？分析 margin、熵、KL 和 logits，不只看 argmax 改变数。
3. 同根完整 4 秒固定执行排名，与首步后重规划的后果排名是否一致？候选分数精确不意味着信用分配适合滚动策略。
4. EP 收益是否集中在原本保守场景，而新增 DAC 损失来自另一类场景？这是全局目标与泛化取舍，不能靠一条轨迹解释。

### 第四步：认真检验轨迹切换假设

对退化、改善、无变化场景分别抽样，优先选跨 seed 重复的现象；展示首次 DAC 违规前后的地图、车身轮廓、速度、旧计划后缀、新计划和候选分数。

比较计划时先转换到同一绝对坐标系，对齐**共同未来时刻**；不同决策时刻的局部坐标不能直接相减。定位切换发生在违规前还是违规后，排除“已经驶出路面导致下一步所有候选失败”的倒因果。

对同根对照 parent/online/oracle 首步，分别设计固定 continuation 与自身 continuation，明确两者回答不同问题。A/B 的更新时序、0.5/1 秒执行间隔和奖励 horizon 是三个因素，不要一次全改。现有 interval1 只能支持诊断方向，不能单独证明切换是根因。

### 第五步：学习动态和效率

从有效 report/更新回执按轮次分析 reward 分布、全零/无信号比例、候选安全覆盖、KL、熵、梯度/更新次数、行为策略版本、重前向 logits。字段以实际保存内容为准，缺失字段如实注明。

比较 A/B 是缓解 episode 内策略漂移，还是仅改变访问状态分布；区分 reward 上升、selector 在同候选上的排序改善和评测安全改善。确认累计预算/断点恢复对可比性的影响。

分别报告训练有效墙钟、失败重试、渲染/模型调用、候选模拟评分、评测开销。不要把 904 个短 episode 描述成长时间连续驾驶，也不要把“无需渲染的奖励分支”当作完整闭环训练完全无需渲染。

### 第六步：按证据决定下一轮

先完成上述分析，再给可检验的最小改动。若确实是接口/时序错误，修复后做配对验证；若是固定轨迹评分与重规划回报错配，优先设计小规模同根对照验证，再讨论 continuation-aware 目标；若主要是学习强度/漂移，固定奖励与执行间隔对照更新幅度；若候选安全覆盖不足，明确冻结生成器的上限。

任何新方法单独建版本，保留本轮作为覆盖和更新时序对照。不应先加切换惩罚、改安全权重、改专家 EP，再用同时发生的变化解释收益。

## 8. 固定选模规则与正式评测

p6 三种子平均相对 parent：R PDMS ≥ +0.005，NR PDMS ≥ -0.005；R/NR 的 NC、DAC、TTC 均 ≥ -0.005。A/B 都通过时选 R 增量更大的方法，相同选 A。评测所选方法的全部三个种子，不看正式测试后挑最好种子。

正式四表为 `ol_navtest`、`ol_rare`、`cl_NR`、`cl_R`。开发集和正式测试必须区分，未完成精确覆盖不能称“完整评测成功”。当前 A p6 不满足安全门槛，B p6 尚缺完整证据；不要提前宣布任何方法最终胜出。

如果两法都不通过，按原规则报告没有方法合格，先完成原因分析，不降低门槛制造成功。若为科学诊断需要评测未通过的模型，应另行明确其用途，不能登记为原规则选中的最终方法。

## 9. 如确实缺少两项开发评测，使用哪个入口

本次交接没有启动任何训练/评测。新会话先排查已有输出；确认缺失后，已有恢复入口会自动预检并复用已验收结果：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930/storage_recovery_20261001/start_recovery_h100.sh eval
```

注意这不是“只跑两个 job”的硬编码命令：它领取就绪评测，开发门槛通过后可能继续正式四表。使用前读恢复文档，确认 session 时钟、锁和当前选模状态；若只打算补两项，应先查看入口支持的任务选择方式。原 eval 恢复窗口为 24 小时，跨日可能过期，不要随意在每台节点生成不同 session 绕过失败登记。

`CHECK_ONLY=1` 仅预检。`QUEUE_IDLE_EXIT` 只代表此时无可领取任务。运行脚本前台输出；不要再把带后台 nohup 的多行命令粘进平台入口。

## 10. 存储清理背景与保护范围

本轮刚完成与论文主线无关的历史产物清理，报告见 `research/storage_cleanup_20261001/retired_postp1_20261001/REPORT.md`。最新这一批移除旧 MID/P1/CL_NLL 传感器图片约 230.502 GiB，清理记录中的可用空间约 626 GiB；此数是当时快照，不代表新会话实时配额。

当前在线训练/评测/checkpoint、V2/V3 主线和消融所需资产已保护。C 消融尚未完成，用户计划重启，预计还需约 200G；C/D 当时均未运行，但停跑不代表可以删除其依赖。后续不要把新腾出的空间全部分配给在线评测图像，也不要因本文提到旧版本就继续删除历史证据。

## 11. 新会话应交付什么

建议新建独立分析目录，输出：

- 完整性审计及模型/场景/补丁来源索引，明确 B_s2 p6 缺口是否已补。
- 三种子 p3/p6 × R/NR 完整比较表、逐场景配对表、按 origin log 的区间。
- 安全退化与进度收益分解、跨 seed 重复失败分类，覆盖改善和未变场景。
- 训练动态与耗时报告；对轨迹切换、候选上限、奖励 horizon 错配分别给支持/反证。
- 最终报告将“已验证事实”“合理假设”“仍需实验”分开，给出下一轮最小对照及继续/停止条件。

可以直接复制给新会话：

> 请先阅读 `/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/1001_online_closedloop_analysis_handoff.md` 及配套 registry_snapshot.json。我们要分析 V3 在线闭环 A/B 各 s0–s2 的结果。六组训练均已到 904 episodes；先核对交接中 B_s2 p6 两项评测未登记的差异，不重复训练。请保留当前 reward 和评测标准，从完整性、跨种子/跨场景效果、安全与进度取舍、更新时序、固定候选评分与重规划后果的差距入手，寻找根本原因。先用已有证据分析，必要时提出有明确区分力的补充实验；不要只据单个失败场景加奖励项，也不要假定加入 simulator 必然提升。分析文件放新目录，不修改冻结实验或已有结果。C 消融还要使用存储，继续保护其依赖。
