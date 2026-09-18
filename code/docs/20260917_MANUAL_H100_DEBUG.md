# 手动四卡 H100 调试

当前不使用 SSH，不运行学长的 remote.sh/setup.sh。先把旧 `run_h100_full_eval.sh` 留在停止状态；保留已验收闭环结果和中断现场。六个新任务通过调试后，再处理旧评测的中断。

在空闲的四卡 H100 实例上执行：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-paper/code
bash debug_paper_tasks_4h100.sh
```

终端会打印 `DEBUG_ROOT=...`，其中 `logs/` 和 `summary.json` 都在共享目录可见。诊断目录使用代码版本摘要命名，代码修复后重新运行会写新目录，旧现场保留。相同版本的成功短训练可复用；未完成模拟器条件仍需检查现场，不能默认覆盖。

检查顺序：B 全部反馈目标四步训练；C 全部结构四步训练；F 三个来源四步训练；D 标签提取、两种 selector 四步训练以及 L2/BC 四卡 DDP 的 8 次 microbatch；A scalar/D4 seed0 的 NR/R bridge 与 scalar 的正式 rare288 开环路径；E 各模式四个历史状态的 sentinel 和候选 0 分支执行。缓存任务每卡一个，模拟器条件顺序执行。只选择候选 0 是验证传输链路，不是 full20 科研结果。

训练和模拟器的检查耗时不同，A/E 会比缓存短训练长。终端显示每步日志路径，一项失败会被记录，随后检查其他项。仅全部检查通过才写 `PASS_DEBUG_CHECKS`；它不表示正式 3 seed 结果完成，也不代替新模型训练后的导出与实际闭环验收。

已修正共享入口的裸 python 依赖、全局盘配额问题、运行时复制 core dump、报错状态覆盖原始异常。任务输出配置在 `../registry/paper_storage.json`，实际落在 HDD2。E 现在共享同场景集合的序列化文件，不再为 20 候选重复保存相同场景。评测 CPU 分配按实例可用核数确定。

代码侧测试已覆盖磁盘不足、精简运行时、启动环境、单项失败仍汇总全部检查等行为。四卡 DDP 和模拟器链路以目标实例此次实际输出为准。

H200 实例也可执行相同命令。六队列入口接受 H100/H200，记录实际型号，继续执行 sm_90 CUDA 扩展检查及原有 bridge 一致性检查。训练超参数保持原配置。旧评测驱动的 H100 限制保持原样，后续处理旧任务时另行验收。

最新验收：`20260917_paper_debug_0ae2ee9554` 在 4×H200 上六项全部通过，约 31 分钟。详见 IMPLEMENTATION_VALIDATION.md 最后一节；可进入六队列正式实验，旧 full eval 的中断另行处理。
