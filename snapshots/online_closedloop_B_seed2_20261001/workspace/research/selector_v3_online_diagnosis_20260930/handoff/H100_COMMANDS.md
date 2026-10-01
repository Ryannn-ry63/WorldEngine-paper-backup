# 8×H100 并行运行手册

新版本目录：

```bash
WORK=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_20260930
```

每个分布式任务分配**单机 8 张 H100**，在不同节点运行下列入口。这里的并行是多台独立 8 卡节点；不要把多个训练作业放到同一组 GPU。无需另外 `torchrun`，脚本会启动单机 8 rank。无需 `nohup` 或手工重定向，终端会直接显示日志和报错，同时保存日志；分布式平台任务命令保持前台运行。

代码与候选清单已预注册，当前未启动正式 GPU 作业。`CHECK_ONLY=1` 只核对导入/登记；技术准入完成后，训练的 CHECK_ONLY 还会验证最终 manifest 和场景资产。

## 1. 启动前检查

```bash
WORK=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_20260930
CHECK_ONLY=1 bash "$WORK/handoff/start_job_h100.sh" admit_0
CHECK_ONLY=1 bash "$WORK/handoff/start_job_h100.sh" train_B_s0
```

最后出现 `CHECK_ONLY` 且退出码为 0 即导入/登记检查通过。尚未完成准入时，不会声称已验证 233 个场景的实际渲染。

## 2. 四台节点分别跑场景准入

节点 1：

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_20260930/handoff/start_job_h100.sh admit_0
```

节点 2、3、4 分别把末尾换成 `admit_1`、`admit_2`、`admit_3`。每台 8 个独立冻结策略进程，覆盖自己的场景分片，每个场景实际渲染并完成 8 次 full20、4 秒响应式候选评分。此阶段没有优化器。

静态候选为 train 150 / development 83。全部技术准入完成后自动生成 `registry/scene_manifest.json` 和完整开发集材料。碰撞、出界、低奖励不是排除理由；程序/资产/数值错误会留下失败记录，不能当作低分场景偷偷跳过。

## 3. 六台节点分别跑 A/B × 三种子

六个任务 ID：

| 方法 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| A：每步更新 | `train_A_s0` | `train_A_s1` | `train_A_s2` |
| B：冻结采集 8 步后逐组更新 | `train_B_s0` | `train_B_s1` | `train_B_s2` |

每台节点使用一个 ID，例如：

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_20260930/handoff/start_job_h100.sh train_A_s0
```

可以提前提交；它会等待准入 manifest。若 150 个 train clip 全部通过，每组训练到 **904 episodes**，在 **456 / 904 episodes** 发布覆盖 3 / 6 遍的不可变检查点标记。若有经核验的技术排除，预算按 `8*ceil(p*N/8)` 重算。

A/B 都从同一离线 V3 父模型开始；不从 online64 接着训练。每个 round、每个 rank 都有 8 次优化尝试，B 只改变采集与更新的顺序。学习率、KL、奖励、候选数、评分时长和主环境 0.5 秒步长相同。

## 4. 多台评测/诊断节点运行同一个队列入口

建议额外启动 4 台或更多 8 卡节点，每台运行：

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_20260930/handoff/start_eval_worker_h100.sh
```

节点通过文件锁领取不同的就绪作业：

- 父模型、online64 的完整 83 场景 R/NR 开发集评测。
- 0.5 秒与 1 秒更换计划的独立诊断，环境仍每 0.5 秒推进、观测和模型前向仍保持连续。
- 48 个代表性根状态、144 次首步后闭环延续；额外覆盖开发集掉分/安全退化场景，单独记录。
- 每组训练 3 / 6 遍检查点的完整开发集 R/NR 评测，不阻塞训练继续采集。
- 完成 A/B 六组最终开发集后，按预注册规则选方法；通过才启动父模型一次及所选方法三种子的正式四表。规则未通过不会自动延长训练或修改奖励。

如希望单独保留诊断节点：

```bash
QUEUE_GROUP=diagnostic bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_20260930/handoff/start_eval_worker_h100.sh
```

`QUEUE_GROUP=evaluation` 则只领取评测任务。队列会跳过已完成和已失败的任务；失败任务需要按原 ID 显式重新运行，避免后台无休止重试。

## 5. 查看进度

不需要 GPU 的终端：

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_20260930/handoff/watch_experiment.sh
```

主要文件：

| 内容 | 位置 |
|---|---|
| 总报告 | `registry/REPORT.md` / `registry/summary.json` |
| 任务定义 | `registry/experiment.json`（63 个预注册作业） |
| 每个任务状态 | `registry/jobs/<任务ID>/status.json` |
| 终端完整日志 | `registry/launcher_logs/` |
| 每个 GPU 的冻结探针日志 | `registry/jobs/<任务ID>/attempt_*/lane*.log` |
| 训练进度 | `registry/jobs/train_A_s0/training.progress.json` 等 |
| 3 / 6 遍覆盖检查点 | `registry/jobs/train_A_s0/training.coverage3.json` / `.coverage6.json` 等 |
| 诊断汇总 | `registry/diagnostic_summary.json`（48 根诊断完成后生成） |
| 冻结方法选择 | `registry/selection.json` |

## 6. 时间预算与恢复

第一项非 CHECK_ONLY 作业启动时，写入全实验 **18 小时墙钟预算**。它不是累计 GPU 小时。单项训练仍有 12 小时上限，并尽量在完整 DDP round 边界停止。调度等待、准入、训练和评测都计入实验墙钟时间；这些只是上限，**不是完成时间保证**。

如果 18 小时内没有完成全部 6 遍结果，但 A/B 三种子都完成了 3 遍的完整评测，汇总只报告同预算 3 遍对照，明确标记 6 遍未完成，不自动当作正式最终结果。

训练意外中断后，使用**同一个 ID**重新运行同一命令即可从最后提交的全 rank 边界恢复；未提交的 round 会新建尝试目录。已完成作业不会覆盖原结果。不要编辑已运行版本的代码、清单、`campaign_clock.json` 或删掉失败/恢复记录来强行继续；若需修代码，应创建新的冻结版本。

若出现 `correctness_failure.json`，表示状态恢复、评分或模型一致性硬检查失败。其余训练将在边界停止；低分本身不会触发这个全局停止。

正式四表的精确覆盖目标：navtest open-loop 12,146；rare open-loop 288；closed-loop NR 288；closed-loop R 288。仅进程退出成功不足以验收，还要求 token 无缺失、无重复、无额外项及有效数值。
