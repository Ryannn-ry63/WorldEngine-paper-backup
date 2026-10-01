# 本地验证记录

状态：实现和本地验证通过；H100 全场景准入、训练、开发集与正式四表均尚未执行。

- 最终 AlgEngine 启动环境下：59 个测试通过。包括真实两进程 Gloo 同步、B 的八步固定策略和逐组重算、无信号跳过、恢复与覆盖检查点补发、计划后缀复用、作业锁/临时目录隔离和 CSV 精确覆盖。
- 两套 Python 启动导入通过。准入和训练 CHECK_ONLY 通过导入/登记阶段；最终全场景 manifest 尚待 H100 准入，不声称已通过全场景预检。
- 真实地图 CPU 回归：同一真实场景、响应式交通、20 条解析工程候选，串行与 2 个分支 worker 的首步/末步哈希和奖励一致；独立逐步执行、倒序恢复全部通过。
- 上述两种执行方式共 40 条轨迹，用实际 legacy MetricManager 同轨迹双评分；对齐动态量后的最大绝对差为 0.0（门槛 1e-6）。原 legacy 动态量差异单独记录。
- CPU 正确性回归总耗时约 597 秒，包含大量逐帧完整状态审计，不代表 H100 训练吞吐量。
- 旧版 657 个源文件哈希未变。最终登记为 63 个独立作业，静态候选 train 150 / dev 83；未开始 18 小时实验计时。

准确的机器可读状态见 [implementation_status.json](../registry/implementation_status.json)，真实地图报告见 [report.json](../registry/local_cpu_parity_final/report.json)。

GPU renderer、全部 233 场景、8 rank NCCL 和完整效果评测的验收仍需 H100 作业；不能将本地通过解释为已取得 PDMS 提升。
