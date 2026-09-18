# 平台项目切换后的 B–F 恢复

用户确认仅切平台项目/算力归属，代码、数据、结果共享路径不变；B–F 已手动中断。恢复保持原 run ID，不修改冻结 paper_study 代码、模型、训练超参数或已验收结果。命令在 `../RESUME_PAPER_BCDEF.txt`，五个任务分别使用四卡实例，建议原 H100 型号、至少 36 可用 CPU 核及相同软件环境。D 的原入口检查已完成缓存训练的 GPU 型号/torch 版本，改变型号可能触发身份检查。

检查时：

| 队列 | 已完成 | 恢复阶段 |
|---|---|---|
| A | attempt_status 为 COMPLETED_REQUESTED_STAGE | 本次不重启；不表示旧闭环已全部完成 |
| B | 39 个正式训练与 native 导出；83 个闭环验收（含 bridge） | --stage eval |
| C | 24 个正式训练与 native 导出；51 个闭环验收（含 bridge） | --stage eval |
| D | 6 个 selector 正式训练与导出 | 原 all 队列，继续 planner |
| E | NR 21/21 集合、R 15/21 集合已有 collection_audit | 原队列，复用已审计集合 |
| F | 9 个正式训练与 native 导出；24 个闭环验收（含 bridge） | --stage eval |

所有检查到的 B–F queue/active 锁，以及 D planner_l2_seed0 train.lock 均可获取；代码哈希与各自 manifest/mechanism 完全相符。

B 的 `common_R_dense_no_feedback_kl_seed0`、C 的 `full_R_feature_only_seed0` 有中断产物且无 audit/accepted。持有 queue/active/prepare/condition 锁后，核验 collection ID 与 contract，把这两个目录改名保存至各自 eval run 的 `archive/manual_project_switch/`，附 archive_reason.json。仅这两项整项重跑。其余已验收结果不变。

F 中断于 `open_navtest_paired_common_seed1`，原开环驱动会保留旧 attempt 并新建一次执行。E 中断于 `full20_1_R_a14`，原分支驱动会保留此集合旧执行产物并整项重跑；已审计集合复用。

D planner L2 seed0 日志最后为 7450/153600 micro-iterations。原保存间隔 38400，该目录没有 iter checkpoint 或可用 latest.pth。因此约五小时 planner 进度不能恢复，须从头运行该模型，不能使用 smoke 的 iter_8 充作正式 checkpoint。已有 selector 训练不丢失。本次没有改变冻结训练协议；planner 保存间隔过长是后续可靠性改进事项，不应再宣传这一部分支持任意时刻无损恢复。

用户另澄清旧 H200 评测也是手动结束、终端输出不可取；此次不继续追查该终端输出，也不将其归因于平台回收。
