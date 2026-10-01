# 冻结版本与报告边界

- 已验证版本：B seed 2 p6。p8 为待运行的 continuation，不预先声称提高。
- 所有 B seeds 的 p6 实体备份：代码目录之外 `releases/B_p6_20261001/`，包含每 seed 8 个完整 checkpoint、selector、完整评测模型、父模型和校验清单。权重不入 Git。
- 当前 B seed 2 p6 完整模型 SHA256：`bc08878d78580530c9d492b8499ff8af14cdc09c5433ab3506f426f2dac49e55`。
- B seed 2 p6 正式 R/NR 提高，但 open-loop 与 DAC 存在退化；不应概括为所有指标全面提高，更不能把挑出的最好 seed 代替三 seed 总体结果。

| 正式表（百分制） | 父模型 | B seed 2 p6 | 差值 |
|---|---:|---:|---:|
| navtest open-loop | 87.54 | 86.39 | -1.16 |
| rare open-loop | 68.02 | 66.09 | -1.93 |
| closed-loop NR | 77.64 | 79.90 | +2.26 |
| closed-loop R | 76.77 | 79.01 | +2.24 |

训练源代码整理纳入 B seed 2 实际已使用的浮点一致性容差 `atol=5e-5, rtol=1e-4`，没有修改 loss/reward 方程。本次只允许恢复/目录/运行编排变化；旧 research 实验源、selection 与评测结果不重写。父模型正式表复用要求另附源码与配置等价审计；p8 新结果仍须完整覆盖验收。

回退使用 p6 manifest 指定的实体权重与其 source provenance；只切换 Git 代码不能恢复 optimizer/RNG。旧工作目录整体归档，保留原 .git 与未提交源码。当前 source 分支保留上游 ancestry，以后可整理为 upstream PR；此次备份只推私有仓库，不修改公开 fork 或 upstream。
