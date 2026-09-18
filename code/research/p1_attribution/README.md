# P1：第二阶段归因对照（独立分支）

状态：实现与 CPU 验证阶段；**未启动正式 GPU 训练或闭环采集**。
基线是已保存的 P1 快照 `077fbca`，分支为 `research/selector-p1-attribution-v1`。
本轮不修改原 P1、V3、GateV3 模型/结果，不自动进入下一种方法或完整 rare288 评测。

## 本轮回答什么

已有 P1 的效果值得保留，但不能把 pair + winner NLL 本身当成新颖性结论。
先检验：在相同策略访问状态、相同原始候选对、相同第二阶段预算下，
闭环反馈来源与 full20 winner NLL 分别提供多少增益，两者是否有交互作用。
重点同时看 rare 难例改善与新增失败，不能只看总 PDM 分数。

这是**在已经接受过闭环反馈训练的 shared 父模型上的第二阶段条件归因**，
不是从零开始证明闭环反馈不可缺少，也不是独立数据采集重复实验。

| 对照 | 第二阶段反馈标签 | Pair | Full20 winner NLL |
| --- | --- | --- | --- |
| L_Pair | 同访问状态下局部代理标签 | 是 | 否 |
| L_Pair_NLL | 同访问状态下局部代理标签 | 是 | 是 |
| P0 | 原已审计闭环分支结果 | 是 | 否 |
| P1 | 原已审计闭环分支结果 | 是 | 是 |
| Continue | 不使用反馈标签；仍采样这些状态用于 KL | 否 | 否 |
| CL_NLL | 原已审计闭环分支结果 | 否 | 是 |

所有组保留 dense GRPO、相同 V3 teacher KL 和原 selector 架构。
共享 256 状态、每状态原来的两个候选、同一 dense cache、500 次第二阶段更新。
每个 seed 从对应 shared 父节点恢复模型、AdamW、NumPy/Torch/CUDA RNG；
不是从最终 P1 出发多训练 500 步。每批 dense16 + feedback16，NR/R 各8，
dense common/hard 各8；lr/weight decay=1e-4，KL=.01，clip=10，temperature=1。
不消费 P2/P3 新查询和诊断分支标签，不添加候选，不把未知候选判成坏样本。
loss 分母仍是全部采样反馈状态，不按“有信息样本数”归一化。

P0/P1 共6个模型及32项历史评测，只有父节点、缓存、分段优化配置、
模型哈希、场景、噪声和评测身份均一致才复用；否则停止，不悄悄换对照。
新增4组×3 seeds=12个训练任务，四卡按同一seed并行四组。

## 局部标签的定义与边界

不能直接把 token 对应的日志 ego_state 当作策略已访问状态。
训练专用观察器在实际 pre-action ego 状态下，对当时原20候选运行已有动态 PDM
评分，并只把原来两个候选的标签交给训练。其周围交通未来来自日志轨迹，
评分使用原 PDM 参考轨迹的 pairwise progress normalization，4秒 horizon。
这与真实闭环分支执行回报有多处差异，**不宣称只隔离了 horizon/长短期因素**。

四个观察条件：round1 NR/R 各96状态，round2 NR/R 各32状态。
访问策略仍为原 V3 / shared_seed0；观察器绝不替换动作。
每个目标核对原状态指纹、候选、实际选中动作；采集后核对全部8次执行决策
的状态/候选/动作和最终结果/首违例时刻。任何不一致均停止，不能丢掉问题状态。
终端12仅发布、不执行，执行决策仍为4..11。

局部标签的有效对数、gap mass、偏好反转，以及训练开始/结束探针的各项加权梯度范数
全部报告。另在报告阶段用 CPU 对六组、三个 seed 的相同父节点和初始批次做无更新
梯度审计，包含历史 P0/P1，记录于 labels/initial_gradient_audit.json。
标签来源不同可能同时改变有效监督量；原始对照不等于监督量匹配证明。
若该方向成立，论文前还需要完整20候选局部监督强基线等后续审查，本轮不偷偷扩展。

## 评测和判断

固定 legacy 开发集 rare58 + common64，NR/R、原噪声
`selector_cfpi_v1_noise0`、LogPlayController、原候选与前向保持一致。
这不是已保存正式 snapshot 使用的完整评测，也不是全新盲测；
common 含训练暴露风险。不得与完整 rare288 成绩混为一谈。

总80项条件：32历史复用 + 48新增闭环评测。
主要对比：P1−P0、P1−L_Pair_NLL、
(P1−P0)−(L_Pair_NLL−L_Pair)；辅助比较 P1−Continue、P1−CL_NLL。
报告全部seed/均值/seed标准差、NC/DAC/TTC/进度、rescued/broken，
并按固定 V3 的 failed / safe-slow / safe-other 分层；不按新方法结果重新定义难例。

先对seed平均，再按log做10000次配对聚类bootstrap，附equal-log敏感性分析。
区间是探索性、未做多重检验校正；跨0不能声称显著提升或已证明不下降。
工程PASS只表示完整性通过；终点始终为 `REVIEW_REQUIRED`，人工复核再决定改进方向。

## 一步步运行

先进入这个新工作树，不要在原 snapshot 或 decision-feedback 分支执行：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-p1-attribution-v1
```

第一步只做 CPU 审计；不会训练或采集闭环：

```bash
bash run_diffusiondrive_selector_p1_attribution_4h100.sh audit p1_attribution_20260913_v1 --gpus 4 --gpu-hours 192
```

拿到四卡后先做硬件预检：

```bash
bash run_diffusiondrive_selector_p1_attribution_4h100.sh preflight p1_attribution_20260913_v1 --gpus 4 --gpu-hours 192
```

确认要开始正式实验后，顺序运行以下命令（或使用下一段一键后台命令）：

```bash
bash run_diffusiondrive_selector_p1_attribution_4h100.sh labels p1_attribution_20260913_v1 --gpus 4 --gpu-hours 192
bash run_diffusiondrive_selector_p1_attribution_4h100.sh train p1_attribution_20260913_v1 --gpus 4 --gpu-hours 192
bash run_diffusiondrive_selector_p1_attribution_4h100.sh eval p1_attribution_20260913_v1 --gpus 4 --gpu-hours 192
```

一键后台运行从 labels 顺序走到最终报告；已审计条件和完整训练任务会复用，
未完成训练每25步存档；失败即停，不放宽一致性标准：

```bash
nohup bash run_diffusiondrive_selector_p1_attribution_4h100.sh all p1_attribution_20260913_v1 --gpus 4 --gpu-hours 192 > p1_attribution_20260913_v1.log 2>&1 &
tail -f p1_attribution_20260913_v1.log
```

只生成进度/报告，不启动GPU任务：

```bash
bash run_diffusiondrive_selector_p1_attribution_4h100.sh report p1_attribution_20260913_v1 --gpus 4 --gpu-hours 192
```

预算192 GPUh是四卡累计上限（最多48小时占用），不是预计时长。
审计会输出基于历史实测的预测；局部评分额外耗时尚未实测。
同一run的代码、预算、输入不能变更。代码更改后使用新run ID，不覆盖旧契约。
未完成闭环条件会整体重跑，旧输出可恢复地移入该条件的 attempt_archive；
只停止当前runner自己创建的进程组，不清理用户其他进程。

产物位于 `experiments/diffusiondrive/selector_p1_attribution_v1/runs/<run_id>/`：
`audit_report.json`、`labels/audit.json`、`train/*/step_500_report.json`、
`attribution_progress.json` 或 `attribution_report.json`、配对场景CSV、累计预算ledger。

## CPU 测试

```bash
/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python -m unittest discover -s research/p1_attribution/tests -v
```

合成测试中的小模型优化器操作不等于正式研究模型训练。
真实缓存预检只做前向/反向，不调用 optimizer.step。
硬件、局部评分与闭环轨迹复现必须在用户后续启动阶段验证；CPU PASS不能替代它们。
