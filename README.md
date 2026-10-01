# WorldEngine selector paper — private recovery checkpoint

最新检查点：**2026-10-01 在线闭环 B seed 2 p6**。见 [当前状态](docs/STATUS_20261001_B_SEED2.md) 和 [版本入口及四表结果](snapshots/online_closedloop_B_seed2_20261001/README.md)。包含冻结源码、恢复补丁、训练端点与全部三种子四表指标；R/NR 总分提升与 DAC 下降同时如实保留。

2026-09-18 的研究备份继续保留：V3 → P1/MID → prefeedback D1–D4 → A–F 补充实验，以及项目迁移后的自有环境入口。

先读 [当前状态](docs/STATUS_20260918.md)、[恢复说明](docs/RECOVERY.md) 和 [同步上游与未来PR](docs/UPSTREAM_WORKFLOW.md)。这是一份可追溯的研究备份，尚不是最终论文开源版。

## 目录

- `code/`：paper 工作区源码、配置、测试、研究实现和运行入口。
- `registry/`：源文件 SHA256、环境版本、运行路径和模型/数据引用索引。
- `reports/`：P1/MID 完整29列结果，以及已验收 D1–D4 rare288结果。
- `evidence/`：历史汇总、逐场景指标、冻结输入/契约、训练端点报告和迁移验收证据。
- `docs/`：当前状态、论文路线、恢复与上游协作说明。

旧文档保留原日期与当时的判断；当前进度以 `docs/STATUS_20261001_B_SEED2.md` 为准。`snapshots/` 保存后续独立研究版本。JSON里的 RUNNING 是历史快照，不能证明任务现在还在运行。

## 两种用途

本仓库 `main` 保留上述干净工作区结构；`research/paper-code-20260918` 保留 WorldEngine 原目录结构和 P1 之前的 Git 祖先，用于比较代码、同步上游和准备未来PR。现有公开 fork `Ryannn-ry63/WorldEngine` 及其上游关系未改变。

这份 Git 备份**不包含**数据集、图像、仿真资产、模型权重、优化器张量、训练cache、Conda环境、CUDA二进制和完整实验输出。仅凭clone无法立即恢复训练。对应文件仍需完成共享存储迁移/备份；路径和哈希用于定位及核验，不是文件本体。私有仓库中的研究备份也不等于已经向学长或公开仓库发出PR。

## 后续同步与贡献

以后可直接让 Codex 阅读 [0918 GitHub / 上游 PR 操作指南](docs/20260918_GITHUB_UPSTREAM_PR_GUIDE.md)，按最新实现整理贡献并通过原 fork 提交 PR。
