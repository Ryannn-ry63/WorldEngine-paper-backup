# 新训练 A/B 模型的补评命令

## 直接启动

建议先提交 12 个独立任务，每个任务申请单节点 8×H100，每个任务都使用同一条命令：

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930/evaluation_20261001/start_eval_h100.sh
```

每个节点通过原 job.lock 领取不同的就绪任务，依次评测。前台显示日志并保存到本目录 logs。已经 COMPLETE 的任务不重复运行，FAILED 任务不自动重试。命令不会训练模型。

已在本地通过登记与依赖预检：22 个新训练模型的开发集任务已就绪，B_s2 的 p6 R/NR 两项等待最终 checkpoint。原基线/online64 和间隔诊断的 8 个评测结果已经存在，队列自动跳过。

## 评测范围与流程

1. 完整开发集 83 场景，A/B × seeds 0/1/2 × p3/p6 × R/NR，共 24 项；p3 为覆盖三遍（456 episodes），p6 为六遍（904 episodes）。
2. 原来的固定开发集选择规则保持不变。全部 p6 三种子结果齐全且方法达到门槛后，自动选择 A 或 B。
3. 仅对合格方法的三种子与 parent 做正式四表：navtest 开环、rare 开环、NR 闭环、R 闭环，共 16 项。若没有方法合格，不自动进入正式测试，也不宣布提升。
4. 原评分脚本、数据、模式、checkpoint 导出校验、精确场景覆盖验收与模型选择规则均复用。

## 多节点与空闲退出

- 可以与 B_s2 续训在不同节点并行。
- 默认 120 秒没有可领取任务就输出 QUEUE_IDLE_EXIT 并退出，避免闲置占卡。它表示没有就绪且未被占用的任务，不表示全部评测完成。
- B_s2 完成或后续正式评测解锁后，可以再次提交同一命令增加并发。仍存活的 worker 会自动继续领取新任务。
- 同一个 session 的所有节点共享同一个窗口；节点重启不会重新计算这 24 小时。

## 独立评测窗口

原训练批次截止时间不改。补评从第一个正式 worker 启动起独立计时 24 小时，clock 记录在 sessions/posttrain_20261001/clock.json。新入口仅进程内替换评测调度与汇总的时钟，原 training campaign_clock.json 不改写；每个评测状态补充 evaluation_window 的路径与哈希。

CHECK_ONLY 不建立窗口，不导出模型，不启动 GPU：

```bash
CHECK_ONLY=1 bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930/evaluation_20261001/start_eval_h100.sh
```

窗口到期后如需继续，明确指定新 EVAL_SESSION，所有新节点使用同一个值；不覆盖先前窗口。

## 结果与校验

汇总仍保存在原实验 registry/REPORT.md 和 registry/summary.json。完整单任务评测必须通过 PASS_EXACT_COVERAGE；不能仅以平台退出码为准。

本地已验证：启动脚本语法、登记/任务依赖预检退出码 0、独立时钟与 evaluation-only 调度两项测试通过、真实汇总函数通过、修复后 B_s1 p6 checkpoint 的全 rank 导出审计通过。未在本地启动正式 GPU 评测。
