# 修复后的 H100 恢复命令

原实验目录和六组已提交检查点不变。所有命令必须使用本目录的入口，不能再使用原目录的 `start_job_h100.sh`。

## 1. 先在一台独立 8×H100 节点运行修复后的 roots48

```bash
REPAIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930

CHECK_ONLY=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash "$REPAIR/start_job_h100.sh" roots48
```

预检退出码为 0 后，在同一台节点运行正式诊断：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash "$REPAIR/start_job_h100.sh" roots48
```

它只运行冻结策略诊断，不更新模型。修复后的诊断会在同一场景中恢复同一个初始完整 simulator snapshot 和静态地图包，再执行根 0/4 的反事实延续；严格状态哈希和候选哈希仍然保留。

## 2. roots48 成功后，释放旧的已确认停止标记

只有当上一步输出 `status: COMPLETE` 后运行：

```bash
REPAIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930
PYTHON_BIN=/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python
"$PYTHON_BIN" "$REPAIR/release_recovery.py"
```

脚本会先检查修复源文件哈希、单卡严格重放、六组原始断点审计和 roots48 成功状态，然后归档旧停止记录，再原子删除旧的 `correctness_failure.json`。如果任何检查不通过，它不会删除停止标记。

## 3. 使用同一个任务 ID 续跑六组训练

在六台独立的 8×H100 节点分别运行：

```bash
REPAIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash "$REPAIR/start_job_h100.sh" train_A_s0
```

将最后的任务 ID 分别换为：

```text
train_A_s1
train_A_s2
train_B_s0
train_B_s1
train_B_s2
```

同一个任务 ID 会从原来已经提交的 boundary checkpoint 继续，A/B 的训练代码、奖励、优化器和 904 episode 目标不变。修复入口会拒绝在 roots48 和释放步骤完成前启动训练。

## 4. 评测队列

释放停止标记后，可以在额外的 8×H100 节点运行：

```bash
REPAIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930
QUEUE_IDLE_SECONDS=120 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash "$REPAIR/start_eval_worker_h100.sh"
```

评测节点两分钟没有就绪任务会自行退出，避免 GPU 利用率过低被平台杀掉。训练到 456 或 904 episodes 发布检查点后，再重新启动同一队列入口即可。

## 5. 查看状态

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_20260930/handoff/watch_experiment.sh
```

修复目录中的日志位于 `logs/`。如果 roots48 失败，不要删除旧停止标记；保存日志后再使用相同的 `roots48` 命令重试。
