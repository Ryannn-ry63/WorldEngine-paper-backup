# 旧完整闭环评测中断与续跑

原任务为 `bash run_h100_full_eval.sh`，run：`20260916_factorial_bridges_4h100_v1`。评测固定 V3 与 D1–D4 共 13 个模型，在 full/rare288、common64 上各运行 NR/R。这是闭环评测，不是训练。

中断于 `full_R_D2_seed2`：simulator 日志在帧和仿真数据写入处明确报告 `OSError: [Errno 122] Disk quota exceeded`。全局用户盘剩余约 11 GB，后续输出改用 HDD2。

已核验 bridge 26/26、full 13/26、common 0/26；剩余 39 个正式条件。已验收结果保留。失败条件没有 accepted/audit，须从该条件开头重跑。

在空闲四卡 H100/H200 实例执行：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-paper/code
bash run_h100_full_eval_resume.sh
```

添加 `--plan` 可只读查看进度。新入口跳过已验收条件，首个待运行条件是 `full_R_D2_seed2`。启动时检查 GPU、至少 36 CPU 核、存储写入和 CUDA 扩展，并在独立目录执行固定 V3 的 NR/R 历史一致性检查。

新输出实际存储在 registry 指定 HDD2 的 `runs/20260917_factorial_full_resume_v1`，引用原 contract、输入、模型和冻结 runtime。已有结果不迁移，不复制大运行时。每项通过原动作审计和身份核验后，在旧 collections 目录建立对应链接，便于旧 status 和后续表格汇总读取。原失败目录在发布已验收替代结果时归档至 `archive/quota_interrupted`，保留内容。再次中断的续跑条件同样保留现场后整项重跑。

启动持有原 queue.lock、active.lock 和新 resume.lock，不终止其他进程。未修改六任务使用的 paper_study 包、旧冻结驱动或原 contract。新旧正式任务应在各自空闲实例运行。

续跑状态：HDD2 目录下 `resume_status.json`；条件日志：`collections/<条件>/logs/`。最终必须 full、common 各 26/26，才报告 `PASS_FULL_AND_COMMON`。

已完成验证：真实旧 run 的只读计划确认剩余 39 项；5 项测试通过，覆盖原 contract 身份、失败现场保留、已接受结果禁止覆盖、未完成结果禁止发布及错误 checkpoint 拒绝。目标实例实际续跑待用户启动。
