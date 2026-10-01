# 磁盘配额故障后的恢复命令

## 当前情况

- A_s0/A_s1/A_s2/B_s0/B_s1：904 episodes 完成，无需重训。
- B_s2：824/904，剩余 80 episodes（10 个八卡轮次）。最后完整轮的 8 个 checkpoint/report 哈希、恢复回执和 selector/reference/optimizer 跨 rank 一致性已核验。
- 评测：8 个基线/间隔任务已完成；22 个已就绪的训练模型开发评测需要重试；B_s2 p6 的 2 个任务等待 coverage6。
- 原训练 campaign 的绝对截止时间已过期。本入口保存原时钟后，仅为 B_s2 恢复窗口延后截止时间；904 episodes、8 steps、奖励、优化公式和累计 43,200 秒训练预算不变。
- 原 12 个评测节点曾重复领取同一任务。本入口对任务、模型导出、开发数据准备和汇总统一使用原子 mkdir 互斥，不依赖原 flock 的跨节点表现。

## 1. 一台 8×H100 续跑 B_s2

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930/storage_recovery_20261001/start_recovery_h100.sh train_B_s2
```

自动先预检，再续跑。沿用已验证的 B_s2 numeric hotfix，从 824 的提交边界继续；失败轮使用新 attempt，不覆盖旧证据。恢复窗口按原剩余计算预算加 600 秒收尾余量生成，第一次实际运行才开始计时；只预检不会延长时钟或修改训练进度。

按历史每轮约 4 分钟估计，剩余训练约 40–60 分钟，视 H100 实际负载而定。最终以 `COMPLETE`、904 和 coverage6 为准。

## 2. 其他 8×H100 节点并行评测

可以与 B_s2 续训同时开始，12 台独立节点使用同一条命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930/storage_recovery_20261001/start_recovery_h100.sh eval
```

本次只使用这个新入口，不同时启动旧评测队列或旧训练脚本。每台节点会先预检，随后领取不同的未完成任务。已完成结果核验后复用；旧失败任务在新恢复窗口内最多尝试一次，再失败会明确报错，不循环重试。

先完成完整开发比较；如果原有固定选模规则通过，继续运行已注册的正式四表。没有方法通过门槛时，不擅自降低门槛或启动正式选模评测。

无可领取任务 120 秒后自动退出，避免空等 GPU。若当时 B_s2 还未产出 p6，则等其完成后重启若干评测节点，仍使用上面的相同命令。`QUEUE_IDLE_EXIT` 只表示当前没有就绪任务，不代表全表完成。

## 存储与日志

- 评测临时图像、frames、merged_ann_files 写入节点本地 `/tmp/worldengine_eval_recovery`，不再大量占用共享配额。
- 数值指标、逐场景数据、执行轨迹、snapshot 记录、配置和日志校验复制至原共享 attempt。仍由原 exact-coverage 验收逻辑判断成功。
- 已验收的本次本地临时目录自动释放。失败时先保存共享日志和诊断，保留本地现场，不把失败记为成功。
- 临时渲染图像不作为持久回放数据保存；本次不删除原有训练/评测输入和历史产物。
- 预检要求本地至少 80 GiB、共享至少 64 GiB 可用。若 `/tmp` 不够，可用 `EVAL_SCRATCH_ROOT=/节点本地大盘/目录` 指定；脚本拒绝与共享 registry 同一文件系统的路径。
- 所有命令前台实时输出，完整启动日志在本目录 `logs/`。原 registry 作业状态和结果路径保持不变。

仅预检：在相同命令前加 `CHECK_ONLY=1`。正式入口已经自动执行预检，无需单独重复。

## 互斥和恢复边界

互斥目录的 `owner.json` 记录主机、PID 和创建时间。进程正常结束或可处理的异常会释放自己的锁；不根据远端 PID 或时间自动抢锁。节点被 SIGKILL 后若遗留锁，需确认原节点已停止后单独处理，不能直接全局删除锁目录。

评测恢复窗口默认 `disk_recovery_20261001`，首次实际启动后持续 24 小时。同一窗口的新失败不会自动再试；修复原因后才用统一的新 `EVAL_SESSION` 启动下一窗口，不要让各节点随意生成不同 session 来绕过失败状态。

本地只做 CPU 断点、导入、存储和多进程互斥测试，没有运行八卡 H100 端到端任务；实际跨节点和 GPU 执行由这些启动命令完成。
