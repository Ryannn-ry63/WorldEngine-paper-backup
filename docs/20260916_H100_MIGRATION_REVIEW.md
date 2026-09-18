# H100迁移验收与历史运行经验：2026-09-16

用户在H100实例完成 `20260916_factorial_h100_migration_v1`。重新执行状态检查，验收文件、audit与collection哈希和模型/场景/contract身份均通过验证。

| 条件 | 场景/决策 | 实际动作审计 | 历史一致性 | 仿真日志时间（UTC） |
|---|---|---|---|---|
| 固定V3 NR | 8/64 | PASS | PASS | 13:52:41–13:58:39 |
| 固定V3 R | 8/64 | PASS | PASS | 13:59:01–14:06:05 |

两种模式均有8份终端发布、0遗漏场景。准备约13:50开始，整个任务约16分钟。证据：

- [NR验收](../../runs/20260916_factorial_h100_migration_v1/collections/bridge_NR_scalar_v3/accepted.json)
- [R验收](../../runs/20260916_factorial_h100_migration_v1/collections/bridge_R_scalar_v3/accepted.json)

目前bridge为2/26，完整集和common集均0/26。只验证了固定V3的迁移；不代表D1–D4正式评测完成。

## 回看旧实现后的结论

旧MID入口和 `research/mid_followup/README.md` 已明确单节点四H100、只启动一个管理进程、不要外套torchrun；通过固定AlgEngine/SimEngine环境及冻结CUDA扩展运行。`run_selector_cfpi.py::_preflight` 还会实际执行MMCV与gsplat的sm_90 CUDA核验，而不只是CPU导入。

新驱动的environment确实复用了旧 `run_selector_cfpi.environment`，双环境和扩展路径没有完全重建。但新驱动的preflight只覆盖CPU导入与配置解析，没有完整搬入旧生产CUDA预检。最初在4090上要求与历史H100逐步一致，以及未按实例CPU配额限制Ray，造成了额外排查。这是迁移安排需要修正之处。

4090同卡24步重复完全一致、对历史H100存在差异；本次H100两种模式均通过。证据支持后续沿用已验证H100环境，不继续为放行正式实验而排查4090复现。GPU硬件不是唯一经过隔离的变量，不能宣称已证明全部差异只来自GPU。

## 后续执行约定

1. 正式对照沿用已通过的H100环境；新实例先核对双Python环境、CUDA扩展和实际CPU配额。
2. 复用旧生产CUDA预检与单入口多卡调度经验；4卡并发仍需显式资源配置。当前迁移脚本只用一张卡，已通过的本次结果无需重跑。
3. 保留已有Ray CPU限制；不能按宿主机可见核数推断容器配额。
4. 新模型仍需各自NR/R原生动作验收；固定V3不重复跑。实现或协议改变另建run，保留历史冻结证据。
5. 开环和29列表格优先接入已验证的旧评测/汇总实现。当前新驱动尚未接入，不能因为状态显示NOT_IMPLEMENTED就理解为原项目没有开环能力。

## 状态输出释义

`attempt_status: null` 表示没有额外的尝试摘要文件；`gpu_lease: null` 表示未通过该驱动管理占卡释放/恢复，均不是失败。`openloop: NOT_IMPLEMENTED_IN_THIS_DRIVER` 描述新闭环驱动的范围。`formal_table_ready: false` 表示全表未完成。`independent_blind_test: false` 表示所用集合不是新独立盲测。`STARTED_NOT_ACCEPTED` 的note是固定提示，不是这次任务的实际结果；本次两个条件都是ACCEPTED。
