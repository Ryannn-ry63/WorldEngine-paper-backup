# 在线闭环 B：代码入口与版本边界

本目录整理的是已经完成 p6 的在线闭环方法，默认 B = `after_rollout8`。算法没有加入新的 DAC 修正。p8 是同一方法从各自 p6 续训两次场景覆盖，结果尚待 H100 运行验证。

- [给学长的代码带读](WALKTHROUGH.md)：按离线 V3 之后的研发顺序阅读。
- [续训与评测](RUNBOOK.md)：三个 seed 独立任务、恢复规则和产物。
- [版本与指标说明](RELEASE.md)：p6 保留、p8 不能自动当作更优模型。

源码保留 WorldEngine 的 `projects/AlgEngine` 与 `projects/SimEngine` 布局及原 Git 历史。当前用户入口是 `scripts/online/run.py`；`innovation3` 下历史 probe 是兼容/验证工具，主线不依赖 research 目录的 Python 注入或 monkey patch。机器配置、数据清单、运行结果和大权重位于代码仓库之外。

本次本地验证见 [VALIDATION.json](VALIDATION.json)，权重索引见 [B_P6_CHECKPOINT_MANIFEST.json](B_P6_CHECKPOINT_MANIFEST.json)。这些文件记录 CPU 检查与既有 p6 证据；不会把尚未运行的 H100 接续标为成功。
