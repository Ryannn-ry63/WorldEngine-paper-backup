# B 三个 seed：p6 接续到 p8

本机只做 CPU 校验。下面三个任务各使用独立节点的 **8 张 H100**，可以同时提交；同一节点不要重复启动。训练后自动运行该 seed 的开发集 R/NR 和正式四表，共六项评测。

```bash
# 任务 1：B seed 0
python /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-selector-v3-online/code/scripts/online/run.py pipeline --campaign /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-selector-v3-online/runs/B_p8_20261001/campaign.json --seed 0
```

```bash
# 任务 2：B seed 1
python /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-selector-v3-online/code/scripts/online/run.py pipeline --campaign /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-selector-v3-online/runs/B_p8_20261001/campaign.json --seed 1
```

```bash
# 任务 3：B seed 2
python /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-selector-v3-online/code/scripts/online/run.py pipeline --campaign /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-selector-v3-online/runs/B_p8_20261001/campaign.json --seed 2
```

命令将自动先运行一个新 DDP round（904 → 912），验收 learner/optimizer/RNG 连续性，再继续到 1200。它是当前续训的第一轮，不额外重复计入训练数据。续训仍是 B，不修改 reward 或 DAC。

将 `pipeline` 改为 `check` 可做 CPU 路径/source/hash 检查；`dry-run` 打印训练命令和评测清单；`train` 只续训；`eval` 对已完成 p8 导出并评测；`report` 重建曲线与分数对比；`verify` 验证完成边界。`verify` 需要 p8 已完成，不能拿预检通过代替 GPU 验证。

`EVAL_SCRATCH_ROOT` 默认 `/tmp/worldengine_online_p8`，必须是节点本地盘并至少空闲 80 GiB；共享盘至少预留 64 GiB。图像和 IPC 临时文件在本地，metrics、plans、日志验收后留在共享结果目录。

每个 seed 输出在 `../runs/B_p8_20261001/B_sN/`：`training.progress.json`、`training.coverage8.json`、`p8/prepared.json`、`evaluation/`、`report/training_curves.png` 与 `report/evaluation.json`。没有可变的 best/latest 链接，p8 不会覆盖 p6，也不会自动取代 p6。

正常中断后重跑同一命令，从最后原子提交轮恢复。任务被强杀时，目录锁可能留下 `*.lock.mkdir/owner.json`；检查记录的 host/pid 与调度任务，确认已结束后再人工清理锁。不要只凭本机 PID 或经过时间抢其他节点的锁。全流程严格校验冻结源码，改算法需新 campaign，不可修改进行中的源码来继续 resume。

B seed 2 历史 p6 记录训练墙钟约 8.13 小时；按 37/113 个 DDP rounds 粗估本次新增训练约 2.7 小时/seed，初始化、机器负载和评测另计。这个估计不是已测得的 p8 时间。
