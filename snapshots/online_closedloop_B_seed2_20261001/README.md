# 在线闭环 B seed 2：2026-10-01 研究检查点

本版本保存离线 V3 rare_tuned 父模型继续进行在线闭环训练后的 B p6 三种子结果，重点记录 seed 2。它是研究备份，不是通过安全筛选的最终方法，也不是可脱离原集群数据与环境直接执行的公开发布包。

## 方法

- 同一个离线 V3 rare_tuned seed 0、epoch 64 父模型初始化所有在线实验。
- A：每次决策后更新；B：固定模型参数采集 8 步，再按时间顺序重新前向、逐组更新。
- 每步 0.5 秒，B 采集期间仍每步重新观察和规划。冻结的是采集期间的参数，不是执行一条 4 秒轨迹而不重规划。
- A/B 使用相同 GRPO exact-group 目标、候选生成器、奖励、学习率 `3e-5`、KL 系数 `1e-3` 和每轮 8 次更新尝试。
- 在线训练使用 150 个场景，p6 检查点为 904 episodes、113 个 8 卡 DDP rounds，覆盖至少 6 遍。
- seed 0/1/2 是在线训练随机种子，评测噪声 seed 统一固定为 0。

## 正式四表结果

16/16 项已通过精确场景覆盖与指标哈希验收。数值为逐场景 PDMS 均值 × 100；不将 CSV 的 average 行再当作场景平均。

| 模型 | Navtest 开放环（12,146） | Rare 开放环（288） | NR 闭环（288） | R 闭环（288） |
|---|---:|---:|---:|---:|
| 父模型 | 87.543 | 68.017 | 77.640 | 76.774 |
| B seed 0 p6 | 85.379 | 66.428 | 76.768 | 77.193 |
| B seed 1 p6 | 84.862 | 64.757 | 76.461 | 76.857 |
| **B seed 2 p6** | **86.386** | **66.088** | **79.896** | **79.014** |
| B 三种子均值 | 85.542 | 65.758 | 77.708 | 77.688 |

seed 2 相对父模型 NR +2.256、R +2.240 分，但 DAC 分别下降 3.819、5.556 个百分点，两项开放环也下降。三种子均值 NR +0.069、R +0.914 分。未由这些数据证明统计显著或跨种子稳定收益。原开发集安全选模结论保持 `NO_METHOD_QUALIFIED`。

seed 2 是事后观察到的较好闭环结果，报告时需保留全部三种子及开放环结果。若后续利用这批正式评测研究 DAC，应说明其开发用途，不能将其继续描述为完全未参与开发的最终测试。

完整表：[逐种子分析](workspace/research/selector_v3_B_p6_formal_diagnostic_20261001/SEED_COMPARISON.md)、[JSON 汇总](workspace/research/selector_v3_B_p6_formal_diagnostic_20261001/summary.json)、[CSV 汇总](workspace/research/selector_v3_B_p6_formal_diagnostic_20261001/summary.csv)。

## 归档结构

`workspace/` 保持原 WorldEngine 工作区内相对路径，各文件与原件逐字节一致：

- `research/selector_v3_online_diagnosis_20260930/code/`：实际使用的冻结训练、仿真、评测源码及配置。
- 同目录 `handoff/`、`registry/`：入口、契约、A/B 覆盖标记、B seed 2 完整进度和最终八个 rank 报告；未打包每一步完整训练轨迹。
- `research/selector_v3_online_diagnosis_repair_20260930/`：重放/恢复源码、数值审计容差 overlay 和本地临时盘评测机制。B seed 2 经磁盘中断恢复，不能宣称它从未中断。
- `research/selector_v3_B_p6_formal_diagnostic_20261001/`：四表入口、完整 16 项结果 JSON、指标 CSV、配置与命令记录，以及启动检查修复。
- `research/online_closedloop_analysis_20261001/`：此前开发集诊断脚本、表格和生成图；图与旧分析的适用对象以报告说明为准，不将 A 的案例图冒充 B seed 2 证据。

## 恢复与运行边界

先运行 `python verify_snapshot.py`，验证所有源码/证据文件 SHA256，以及四表场景覆盖和分数。

模型引用见 [MODEL_REFERENCES.json](MODEL_REFERENCES.json)。本备份不含权重、优化器张量、场景 pickle、图像、地图、仿真资产、完整轨迹缓存或 Conda/CUDA 环境。GitHub 备份不能替代这些外部文件的保存。

原共享存储仍可用时，真实入口在原工作区 `research/selector_v3_B_p6_formal_diagnostic_20261001/start_h100.sh`；四种 scope 是 `cl_R`、`cl_NR`、`ol_rare`、`ol_navtest`，每个独占 8×H100，顺序评测四个固定模型。已完成的原任务会校验后跳过。命令见[原运行说明](workspace/research/selector_v3_B_p6_formal_diagnostic_20261001/README.md)。

归档保留原绝对路径与不可变契约。不要直接在 clone 后的目录运行训练/评测入口：它们可能引用原工作区的 registry，并且新位置会导致冻结代码路径校验变化。恢复应在隔离目录补齐外部资产，依据本快照建立新的路径配置与版本契约，重新验收；不原地篡改旧实验契约。

## 验证范围

原 16 项评测来自用户运行的 8×H100 节点，本次仅打包并复核，不重复 GPU 评测。归档检查包括逐文件哈希、16 项精确 token 覆盖与 CSV 均值、代码语法及启动路径回归测试。保留旧分析与运行时 SHA256，新增 Git 版本不改变模型、奖励或评测结果。
