# B p6 固定模型四表诊断评测

2026-10-01：评测父模型与 `train_B_s0_p6`、`train_B_s1_p6`、`train_B_s2_p6`，每张表顺序运行四个固定模型，四张表可跨节点并行。所有路径沿用共享存储，需与当前工作区共享相同挂载。每个任务独占一个 8×H100 节点。

## 四个任务

每个节点先进入工作区：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
```

节点 1：R 闭环，288 场景。

```bash
bash research/selector_v3_B_p6_formal_diagnostic_20261001/start_h100.sh cl_R
```

节点 2：NR 闭环，288 场景。

```bash
bash research/selector_v3_B_p6_formal_diagnostic_20261001/start_h100.sh cl_NR
```

节点 3：Rare 开放环，288 场景，官方 NAVSIM 重评分。

```bash
bash research/selector_v3_B_p6_formal_diagnostic_20261001/start_h100.sh ol_rare
```

节点 4：Navtest 开放环，12,146 场景，ADE/FDE 与官方 NAVSIM 重评分。

```bash
bash research/selector_v3_B_p6_formal_diagnostic_20261001/start_h100.sh ol_navtest
```

用平台长任务终端或 tmux 启动，避免断线终止进程。不要在同一个 8 卡节点同时启动四条命令。每条命令自动检查 8 张 H100、运行环境和空间，再顺序执行该表四个模型。每个模型单次最长 24 小时；不是四个模型合计 24 小时。

临时图像和 IPC 写节点本地 `/tmp/worldengine_B_p6_formal`，要求本地至少 80 GiB、共享盘至少 64 GiB 空闲（最低启动门槛，不是占用上限）。如平台另有本地盘，启动前设置 `export EVAL_SCRATCH_ROOT=/本地磁盘路径/worldengine_B_p6_formal`。归档保留指标、轨迹、预测、配置和日志，完整验收成功后删除本次临时目录。失败临时目录保留。

## 结果和恢复

日志：本目录 `logs/`。各模型：`jobs/<table>__<policy>/attempt_*/result.json`，只有精确场景覆盖检查通过才标记 COMPLETE。

查看所有任务，并更新 `summary.json`、`summary.csv`（指标乘 100，三种子均值、样本标准差、相对父模型差值）：

```bash
bash research/selector_v3_B_p6_formal_diagnostic_20261001/start_h100.sh --status
```

重复启动原命令会验证并跳过 COMPLETE；不会自动重试 FAILED。排查失败后在原命令末尾加 `--retry-failed` 重跑，该表之前完成的模型不会重算。进程被强杀留下 RUNNING/锁时，需要先核实原节点进程已退出；本入口不会自动抢占远端锁。

## 评测来源及用途

复用 `selector_v3_online_diagnosis_20260930/handoff/selector_eval_runtime_20260930.py`、冻结四表脚本、`registry/fulltable_data_contract.json`，以及修复入口 `selector_v3_online_diagnosis_repair_20260930/repair_run.py` 的 `evaluation_job` 与原完整覆盖验收。复用 `storage_recovery_20261001/recover.py` 的跨节点 mkdir 锁和本地盘归档机制。

固定评测噪声种子 0、重规划周期 0.5 秒，记录闭环执行轨迹。`manifest.json` 固定四个模型及输入 SHA256。任务结果使用独立目录，不调用旧选模队列或旧汇总器，不更改旧 `NO_METHOD_QUALIFIED` 结论。

这是正式四表数据及评测协议上的诊断比较，用于回答 B p6 是否提升总分并观察 DAC；不意味着 B 已通过开发集安全筛选。报告全部三个种子，不按正式评测分数再挑 seed。若后续利用这些结果改进 DAC，应在论文中说明其开发用途，不能将同一批结果描述为完全未参与开发的独立最终测试。

本机仅做 CPU 契约校验和命令 dry-run；真正 8×H100 GPU 预检在目标节点启动时执行。

2026-10-01 启动修复：`campaign_stop_file` 是可选故障停止标记，缺失为正常；存在时仍阻止运行。旧 `campaign_deadline_file` 不作为本次独立评测的必需输入。已保留修复前入口及 manifest 于 `fix_preflight_stop_marker_20261001/`，并补充 `test_preflight.py` 四项回归检查。此前四次失败均发生在创建任务状态前，直接重提原命令即可，无需 `--retry-failed`。
