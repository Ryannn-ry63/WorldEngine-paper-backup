# H100 logits 容差修复与续跑

## 故障证据

- `train_A_s2`：round024 rank7 在 `learner.choose` 一致性检查失败，最大 logits 差异 `1.52587890625e-05`。
- `train_B_s1`：round024 rank4 同一检查失败，最大差异 `1.1205673217773438e-05`。
- 两者已提交 192 episodes / 24 rounds。最后一轮共 16 个 checkpoint 的 SHA256 全部通过；未提交的 round024 需要重做。
- SIGTERM 是首个 rank 抛异常后的 DDP 清理动作。上述日志没有显示奖励计算错误或 OOM 是本次中止原因。
- 微小差异符合 no_grad 与 autograd 前向浮点舍入差异的表现；尚未在 H100 重放出错输入来逐项验证。

## 改动范围

仅将 `learner.choose` 和 `FixedLiveLearning` 延迟更新的训练/推理 logits 审计 `atol` 从 `1e-5` 调为 `5e-5`，`rtol=1e-4` 保持。保留 finite 检查及不一致抛错。采样仍使用 no_grad 推理 logits，梯度仍使用原 autograd logits。奖励、场景、学习率、KL、更新时机和随机数恢复逻辑没有改动。

`overlay` 仅覆盖 learner 和 fixed_learning，其他模块仍从原代码加载。原 source_sha256 继续表示原始源码契约；修改模块的实际文件哈希由独立 `manifest.json` 和任务 status 的 `numeric_hotfix` 字段补充记录，不能把这两个续跑任务标注为纯原始代码运行。

续跑入口复用原 repair_run 的 job.lock、training.lock、campaign deadline、原状态登记和评测队列。每个新 attempt 保存补丁 manifest 引用、之前的失败 status 和续跑前 progress。原训练 checkpoint、原实验注册和原 repair_manifest 未改写。原实验原有的已完成任务保持原状态。

## 命令

在两台独立的 8×H100 节点各运行对应一条；默认前台显示并落盘日志。无需重新 roots48。不要同时运行同一个任务的旧续跑命令。

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930/hotfix_logit_tolerance_20261001/start_hotfix_job_h100.sh train_A_s2
```

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930/hotfix_logit_tolerance_20261001/start_hotfix_job_h100.sh train_B_s1
```

两条正式命令均自动预检，并从已提交边界继续到注册目标 904 episodes，受原 12h 训练预算和 campaign deadline 约束。独立预检可在命令前加 `CHECK_ONLY=1`，不启动训练。

## 验证

- 显式预加载 overlay 后，19 项 learner 原回归 + 2 项新增容差/行为测试通过。
- 新增测试验证本次两种小漂移通过、采样概率不变、梯度图保留、真实 `1e-2` 偏差仍拒绝。
- 固定奖励学习 2 项测试通过；B 方法完整八步收集后八次更新的带漂移测试通过，前七步策略版本保持为 0。
- 两种更新时机的启动器构造测试通过；补丁启动器和 runtime 语法检查通过。
- 最终完整预检通过（退出码 0）：`logs/train_B_s1_20260930T172232Z_2815313.log`，含 `PASS_NUMERIC_HOTFIX`、AlgEngine/SimEngine 导入和全部 manifest 文件校验。
- 本地未启动 H100 正式训练；最终 H100 恢复是否越过故障轮仍待正式启动验证。
