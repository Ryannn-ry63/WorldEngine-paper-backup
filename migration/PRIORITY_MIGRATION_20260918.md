# 必要文件优先迁移

用户要求优先迁移必要文件，完整保留全部内容，避免并行写入风险。本次调整只改变复制顺序，不删除源/目标、不使用 --delete 或 --inplace，不改变冻结模型和实验实现。

## 当前已实施

1. 用真实 rsync 小样本验证：中断留下完整文件后的续传、图片/逐帧数据最终保留、软链接保留、目标额外文件不删除、同时第二个复制器被锁拒绝、同大小同时间戳内容损坏被 checksum 检出。
2. 切换前检查所有已有 queue.lock/active.lock 未占用；正常 SIGTERM 退出旧复制进程组，等待四个旧进程全部退出后确认 copy.lock 释放。
3. 启动单个复制器 `priority_results_copy.py`，继续原目的地，复用已完成文件和 .rsync-partial。
4. 第一批优先复制 ABCDEF v3 的 76 个选定路径：根清单/状态/契约及 native、checkpoints、runtime、configs、data（队列自己的训练缓存，不是公共dataset）、scenarios、preflight。不会把公共 data 软链接目标复制进来。
5. 第一批完成后做 rsync checksum dry-run；所有选中的原生模型和 selector/optimizer checkpoint 再逐个对照源/目标 SHA256。成功标志为 `critical.accepted.json` 中的 PASS_PRIORITY_BYTES_NOT_RESUME。
6. 随后逐个补齐 ABCDEF 当前运行/评测目录，包括被审计读取的图片、逐帧文件、日志与失败现场；完成每个目录立即 checksum 校验并记录独立验收。再补 reports 和全部旧恢复/调试运行目录，最后对完整 runs/reports 做只读 checksum 校验。

没有丢弃“低优先级”内容：旧复制正在处理的 factorial_full_resume_v1 已保留，调整到当前六队列之后继续；旧调试和失败记录也最终复制。最终入口仍保持 PASS_BYTE_COPY_ONLY，旧启动器全量校验门槛未放宽。

## 为什么还不直接放行六队列

根模型/配置完整不代表已经完成闭环结果的复用审计。发现旧驱动会检查逐帧记录与其他来源，所以完整队列目录优先于按后缀粗略排除图片。任何优先批次通过都不被当作完整续跑验收。

必要文件先到后，可以提前核对模型和准备续跑。若要在全量复制结束前启动某个队列，还必须完成该队列全部来源依赖和既有条件复用验收，并把其目录从后续写入范围中彻底移出；在这之前不开放运行。当前仍要求源/目标实验暂停写入。data 继续由学长负责，与此次复制排序无关。

## 查看进度

- 主状态：`priority_copy_20260918/priority_copy_status.json`
- 优先清单：`priority_copy_plan_20260918.json`，实际 NUL分隔清单 `priority_copy_20260918/critical.files0`
- 每批复制/校验：`priority_copy_20260918/<name>.copy.log` / `.verify.log` / `.accepted.json`
- 启动日志：`priority_copy_launcher.log`；进程 PID 文件：`priority_copy_pid.txt`
- 旧进程切换记录：`priority_copy_switch_20260918.json`
- 冻结新复制器实现哈希：`priority_copy_implementation_20260918.json`

不要根据状态文件的 COPYING 单独判断进程存活。失败时保留日志和复制内容，检查原因后恢复；同大小文件损坏会停止，不静默接受。完整复制和校验通过前，不承诺“所有文件已完好迁移”。
