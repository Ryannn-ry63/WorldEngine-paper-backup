# 提前进行单卡迁移续跑验证

## 运行

在仍能访问原公共 data 的 H100/H200 实例（至少10个可用CPU核）执行：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-paper/code
bash run_owned_resume_probe.sh 0
```

脚本自动重新检查该实例的自有环境/GPU，然后验证旧结果复用及新闭环。无需等所有历史文件迁移完成。先前完整迁移入口 `run_owned_runtime.sh run` 的门槛保持不变，不用此诊断脚本启动任意正式队列。

## 范围与判定

- 要求已迁移 A_v3 完整目录具有 checksum 通过证书；执行时再次检查实际输入、模型、冻结代码及选中的旧结果哈希。
- 读取 A 的 D4_seed0 / open_rare 旧结果，用原审计函数校验其 288 个 token 的指标、契约与产物。
- 复制冻结的 871 个运行代码文件至自有独立目录，每个文件对照原记录 SHA256。data 只保留原软链接，不复制公共数据。
- 仅新建诊断资源契约：单GPU、8场景桥接、NR和R各一组。使用原模型、数据、噪声及审计算法；新结果与历史 D4_seed0 对照指标、输入、logits、候选选择和动作。
- 每组通过后再次调用原 collection 函数，执行完整审计并复用结果，确认不用重新计算且 audit 内容未变。
- 输出位于 `WorldEngine-owned/runtime/resume_probes/owned_resume_probe_时间_PID/`，位于后台复制目的地之外；正式 A/B 及其他队列不写入。没有硬链接共享写入。
- 通过标志：该目录 `summary.json` 为 `PASS_ISOLATED_RESUME_PROBE`。失败时保留现场，日志记录当前阶段。重新运行命令会创建新的诊断目录。

该验证代表单卡闭环评测与已完成结果复用，不表示四卡DDP、ABCDEF全部条件、训练optimizer续训或新项目data挂载已经验收。数据迁移后仍需核对新链接/绝对路径、输入哈希并在新项目做小样本复测。旧资源路径仅在本进程及子进程中映射，实际非data依赖来自global_user；不改变宿主机挂载或学长目录。

## 本地预检

CPU审计日志：`migration/owned_resume_probe_cpu_audit.log`。该阶段复用旧结果并检查冻结依赖/规划器/模拟器CPU导入；通过标志为 `PASS_OLD_RESULT_REUSE_AND_CPU_PREFLIGHT`，不是GPU运行通过。

2026-09-18 09:56 UTC 本地实际检查通过：旧 D4_seed0 / open_rare 288 token 结果复用、冻结输入/模型校验、planner 配置导入、simulator 导入。证据：`owned_resume_probe_cpu_acceptance_20260918.json`。H100 GPU 闭环待用户执行上述命令。

## H100 单卡实际闭环验证已通过

`owned_resume_probe_20260918T095803Z_133040` 已返回 `PASS_ISOLATED_RESUME_PROBE`，耗时约33分15秒。自有环境/GPU、旧288 token开环结果复用、冻结输入/CPU预检、NR/R各8场景历史一致性、新闭环结果完整审计与复用均通过。复查两组 accepted/audit 哈希匹配，正式源契约和inputs未变。证据：`owned_resume_probe_h100_acceptance_20260918.json`。

这是原data挂载下的单卡诊断验收；不等于全量迁移、四卡、训练optimizer续训或新项目data就绪。`formal_resume_authorized: false` 属于诊断范围标记，不是失败。
