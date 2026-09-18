# The Candidates Are Already There

## Reward-Driven Residual Selector Post-Training for Diffusion-Based Driving

中文暂定标题：

> **更优轨迹已在候选之中：面向扩散式自动驾驶的奖励驱动残差选择器后训练**

## 1. 核心问题

DiffusionDrive 会为一个场景生成 20 条候选轨迹，但部署时只能输出 selector 排名最高的一条。候选生成质量和最终规划质量之间因此还隔着一个选择过程。

本文关注的问题不是重新训练感知模块或轨迹生成器，而是：

> 当冻结的扩散规划器已经生成了更好的候选轨迹时，能否通过 reward-driven post-training，让模型更可靠地把它选出来？

论文首先通过 oracle 分析比较：

```text
Base selector 实际选择轨迹的 reward
                  vs.
同一组 20 条候选中的最高 reward
```

如果二者存在稳定且显著的差距，就说明规划性能的一部分瓶颈来自 selection，而不完全来自 generation。本文将这一现象称为 **selection bottleneck**。

## 2. 总体故事链条

```text
Base DiffusionDrive
        ↓
Oracle 分析发现：候选集中经常已有更高 reward 的轨迹
        ↓
核心诊断：生成能力强于最终选择能力
        ↓
V2：保持原模型结构，只后训练原生 selector
        ↓
验证 reward-based selector post-training 是否有效
        ↓
进一步诊断：原生 selector 的输入和独立评分结构形成表示瓶颈
        ↓
V3：冻结完整 base policy，添加零初始化的 scene-conditioned residual selector
        ↓
用相同的 group-relative reward 训练 residual policy head
        ↓
Rare：研究困难场景与普通场景如何组成后训练数据
```

三个阶段分别回答三个问题：

1. **Objective：** reward-based selector post-training 是否有效？
2. **Architecture：** 什么样的 selector 能更充分地吸收 reward 信号？
3. **Data：** 哪些场景最适合用于 selector post-training？

## 3. Base：原生 selector 的评分逻辑

DiffusionDrive 原生 selector 是 `plan_cls_branch` MLP。它对每条候选的 `traj_feature` 独立打分：

```text
candidate traj_feature
        ↓
plan_cls_branch MLP
        ↓
candidate logit
        ↓
在 20 个 logits 上取 argmax
```

原始监督训练中，系统计算各 plan anchor 与 GT 未来轨迹的距离，将距离最小的 anchor 设为正类，再使用 focal classification loss 训练 selector。因此，原生 logit 更接近“该候选与示范轨迹匹配程度”的学习分数，而不是 PDM reward 的直接预测值。

相关代码：

- 原 selector：[diffusion_planning_head.py](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py)
- anchor matching 与 focal classification loss：同文件中的 `LossComputer`

## 4. V2：Parameter-only Selector Post-Training

V2 不改变模型结构。它从 base checkpoint 初始化原生 selector，并执行：

```text
冻结 perception / BEV / diffusion generator / trajectory regression
        ↓
保留同一次 forward 生成的完整 20 条候选
        ↓
只更新原生 plan_cls_branch MLP
        ↓
使用每组候选的 PDM reward 进行 group-relative optimization
```

V2 是最干净的后训练实验，因为模型结构、selector 输入和初始化均来自 base model，唯一核心变化是训练目标从原始监督信号转为 reward 信号。

V2 可以支撑的结论是：

> 在候选生成器完全冻结的情况下，仅通过 reward-based post-training 更新原生 selector，也可以改善最终规划性能。

V2 回答的是“后训练是否有效”，应当作为论文的核心诊断实验和强基线，而不只是一个被 V3 替代的旧版本。

## 5. 从 V2 到 V3：表示瓶颈

原生 selector 存在以下限制：

- 对每条候选独立评分，不能显式比较同组候选；
- 主要依赖单条候选的 final feature；
- 不直接编码最终轨迹的位置、朝向和运动学变化；
- 不能显式读取候选沿线的 BEV 信息；
- 对 ego、status 和 surrounding agents context 的利用受已有 feature 表示限制。

如果 V2 相比 Base 有提升，但与 candidate oracle 之间仍存在明显差距，且随着训练增加逐渐饱和，那么可以形成如下推论：

> Reward 后训练信号有效，但原生 selector 的参数化和可见信息限制了它从 reward 中继续学习。

这一表示瓶颈是引出 V3 的必要桥梁。论文需要用 oracle regret、困难场景分类结果、训练饱和曲线或表示消融提供证据，而不能只凭网络结构作出判断。

## 6. V3：Scene-Conditioned Residual Selector Post-Training

V3 不再更新原生 selector，而是冻结完整 base model，并增加一个新的 residual selector：

```text
冻结的原生 selector
candidate feature → base logits

新增的 scene-conditioned selector
candidate feature + trajectory geometry + route BEV
+ ego/status/agents context + candidate-set interaction
        ↓
delta logits

final logits = base logits + delta logits
selected trajectory = argmax(final logits)
```

新 selector 包含 feature projection、trajectory geometry encoder、route-BEV cross-attention、scene-context cross-attention、candidate-set Transformer 和输出 MLP。它不是对原生 MLP 的小幅结构修改，而是一套新增的策略头。

相关代码：

- [diffusion_grpo_scene_selector.py](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py)
- [diffusion_grpo_online_planning_head.py](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py)

### 6.1 残差机制

残差机制不是重新预测并替换 base logits，而是学习对 base 排序的修正：

```text
base logits:   [2.1, 1.7, 0.4]
delta logits:  [-0.5, 0.8, -0.1]
final logits:  [1.6, 2.5, 0.3]
```

在这个例子中，Base 选择候选 1；residual selector 结合场景信息后降低候选 1、提高候选 2 的相对分数，最终改选候选 2。

新 selector 的输出层采用零初始化，因此训练第零步满足：

```text
delta logits = 0
final logits = base logits
```

这保证 V3 从与 Base 完全相同的决策行为开始。不过，新 selector 的内部参数基本是随机初始化的，因此准确表述应为：

> V3 的行为从 Base 起步，但新增 residual policy head 的参数从头训练。

另外，残差加法本身不保证修正始终很小。如果没有 KL、delta norm 或蒸馏约束，新分支可以输出很大的正负 residual，并在训练后主导最终选择。

### 6.2 V3 的后训练定位

V3 不应表述成“使用 GRPO 微调原 selector”，因为原 selector 被冻结，实际训练的是新增网络。更准确的名称是：

- scene-conditioned residual selector post-training；
- frozen-policy residual head adaptation；
- GRPO-trained residual policy-head adaptation。

V3 仍可归入广义 post-training，理由是：

- perception、BEV backbone 和 candidate generator 完全冻结；
- 原 selector 完全冻结；
- 候选动作集合及其生成过程不变；
- 新分支零初始化，初始行为等于 Base；
- 不重新进行大规模监督预训练；
- 只使用 reward 训练最终候选选择的 residual policy head。

因此 V3 更接近“冻结基础策略上的 residual adapter training”，而不是重新训练完整驾驶模型。

## 7. Reward Objective 的准确表述

V2/V3 使用同一场景完整 20-action set 的 PDM reward，在组内进行标准化：

```text
A_i = (r_i - group_mean) / (group_std + epsilon)
```

基础 V3 的目标为：

```text
L_policy = -mean_groups[sum_i pi_current(i) * A_i]
L_total  = L_policy + beta * KL(pi_current || pi_reference)
```

因为 20 个候选动作全部可枚举，这相当于对完整离散动作集计算精确 expected advantage。它不同于常见的 sampled PPO-style GRPO：正式 V3 不依赖 action sampling，也不使用固定 ratio clipping。

论文中建议使用：

> exact full-action group-relative policy optimization

而不是声称严格复现标准 GRPO。

基础 V3 没有 imitation loss、原始 focal classification loss或 trajectory regression loss。正式 V3 的 `beta=0` 时，新增 selector 实际只由 group-relative policy loss 训练。后续 gate-conditioned 版本额外加入 safe candidates 内部的 quality policy term，应作为独立变体说明。

## 8. 可以与不可以声称的结论

### 8.1 V2 可以支撑

> Reward-based post-training 原生 selector，在不改变候选生成器的情况下可以提高规划性能。

前提是 Base/V2 使用一致的评测协议，并确认只有原 selector 参数发生变化。

### 8.2 完整 V3 当前可以支撑

> 在冻结 DiffusionDrive 基础策略的情况下，新增并 reward-train 一个 scene-conditioned residual selector，可以进一步改善候选选择。

### 8.3 当前不能直接支撑

以下结论在没有严格消融时都过强：

- “V3 的提升完全来自 GRPO”；
- “V3 证明 residual connection 一定优于直接预测”；
- “V3 证明 scene context 是提升的唯一原因”；
- “V3 提升证明模型获得了更强的场景理解”；
- “V3 只是对原 selector 做了微调”；
- “V2 到 V3 是严格单变量比较”。

当前 V2/V3 同时存在 selector 架构、输入信息、参数量、epoch、learning rate 和 KL weight 等差异。未经控制的结果只能说明完整 V3 配方有效，不能将提升单独归因于某一设计。

## 9. 必须补齐的关键实验

### 9.1 主结果

| 方法 | 训练部分 | 目的 |
|---|---|---|
| Base | 无 | 原始基线 |
| Candidate oracle | 无 | 测量 selection headroom |
| V2 | 原生 MLP | 验证 parameter-only reward post-training |
| V3 | residual scene selector | 验证完整 residual adaptation 方法 |

### 9.2 分离“新网络”“残差”和“场景信息”

| 方法 | 输入 | 输出方式 | 回答的问题 |
|---|---|---|---|
| V2 | candidate feature | 原生 logits | 原 selector 后训练是否有效？ |
| Direct new MLP | candidate feature | 直接输出 logits | 从头训练新 head 的基线是什么？ |
| Residual MLP | candidate feature | base + delta | residual initialization 是否有价值？ |
| Direct scene selector | feature + scene | 直接输出 logits | 场景 selector 从头决策的效果如何？ |
| Residual scene selector | feature + scene | base + delta | 完整 V3 是否最好？ |

关键比较：

```text
V2 vs Base
→ reward-based post-training 的作用

Residual MLP vs Direct new MLP
→ residual connection/base initialization 的作用

Residual scene selector vs Residual MLP
→ 场景信息和候选交互的作用

Residual scene selector vs Direct scene selector
→ 保留 base logits 的作用
```

### 9.3 V3 输入与模块消融

建议逐项增加：

```text
feature only
+ trajectory geometry
+ route BEV
+ ego/status/agents context
+ candidate-set attention
```

还需要加入参数量匹配的 large-MLP baseline，否则无法区分提升来自场景输入，还是仅仅来自更大的模型容量。

### 9.4 Residual 稳定性与约束

正式 V3 使用 `KL weight = 0` 时，residual 可能完全覆盖 base 排序。建议比较：

```text
beta ∈ {0, 1e-4, 1e-3}
```

并报告：

- `KL(final policy || base policy)`；
- selection disagreement；
- delta-logit norm；
- common/rare 场景各自的 reward gain；
- oracle regret；
- 训练稳定性和数据效率。

必要时可以研究：

```text
L = L_group_policy
  + beta * KL(final_policy || base_policy)
  + lambda * mean(delta_logits^2)
```

这能把“从 Base 开始”进一步加强为“受约束地修正 Base”。

### 9.5 公平比较要求

关键消融应尽量保持一致：

- train/development/certification split；
- candidate cache 与 reward provenance；
- optimizer、batch size 和训练步数；
- learning-rate sweep 预算；
- noise seeds 与 training seeds；
- checkpoint selection rule；
- 最终闭环/官方评测协议。

## 10. Rare-Original 的位置

Rare-original 不应与 V3 架构共同包装成一个不可分割的方法。它改变的是后训练数据分布：从完整原始 navtrain 中筛出 collision、off-road 和低-progress token，并配对同 log 的 strict-common token，形成 50% rare + 50% common 的训练采样。

建议将其作为第三阶段或扩展实验：

```text
V2：后训练目标是否有效
V3：什么参数化能更好吸收 reward
Rare：怎样的数据分布能进一步改善困难场景
```

它回答的是 data curriculum / hard-example sampling 问题，而不是 V3 residual architecture 是否成立。

## 11. 推荐的论文贡献表述

### Contribution 1: Selection bottleneck

> We identify a substantial selection bottleneck in diffusion-based driving policies: their generated candidate sets frequently contain trajectories with considerably higher rewards than those selected by the native policy head.

### Contribution 2: Reward-based native-selector post-training

> We show that exact full-action group-relative post-training of the native selector improves planning performance while leaving perception and trajectory generation unchanged.

### Contribution 3: Residual selector adaptation

> To overcome the native selector's representation bottleneck, we introduce a zero-initialized scene-conditioned residual selector that learns reward-driven corrections over a fully frozen base policy.

## 12. 推荐摘要故事

> Diffusion driving policies often generate better trajectories than they ultimately select. We first identify this selection bottleneck by measuring the gap between the native selector and the oracle candidate within the same generated action set. We then show that reward-based post-training of the native selector improves planning without modifying candidate generation. However, the native selector independently scores candidates from limited representations, restricting its ability to exploit group-relative rewards. To address this limitation, we introduce a zero-initialized scene-conditioned residual selector that corrects the frozen base logits using trajectory geometry, route-aligned BEV features, traffic context, and candidate-set interactions. The resulting method preserves the base policy at initialization and trains only the residual decision head with an exact full-action group-relative objective.

中文版本：

> 扩散驾驶策略往往已经生成了更好的候选轨迹，却没有在最终决策中将其选中。我们首先通过同一候选集内的原生选择与 oracle 选择差距，揭示扩散规划器中的 selection bottleneck。随后，我们证明，在不改变候选生成过程的情况下，仅通过 reward 后训练原生 selector 即可改善规划性能。进一步分析表明，原生 selector 对候选独立评分且可见信息受限，制约了其从组内 reward 中继续学习。为此，我们提出零初始化的场景条件 residual selector，利用轨迹几何、沿轨迹采样的 BEV、交通上下文和候选集合交互，对完全冻结的 base logits 进行修正，并仅使用完整候选动作集上的 group-relative objective 训练该残差决策头。

## 13. 一句话主张

> **V2 证明 reward-based selector post-training 有效；V3 揭示并缓解原生 selector 的表示瓶颈，使冻结的基础策略能够通过零初始化 residual head 吸收更多后训练收益。**

在完成 residual、scene inputs、model capacity 和训练超参数的严格消融之前，应将后半句表述为待验证的核心假设，而不是已经被单一 V2/V3 分数完全证明的因果结论。

