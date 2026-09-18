# 2026-09-17 实现验证记录

实现路径：`paper_study/`，脚本 `run_paper_study.sh`。命令见 `NEXT_PAPER_COMMANDS.txt`，实验解释与未完成的论文阶段见 `20260917_PARALLEL_EXPERIMENTS.md`。

本机为单张 RTX4090。下面的 GPU 检查没有冒充四卡 H100 验收。未停止占卡程序，验证使用当时剩余显存；没有启动六条正式 H100 队列。

## 已通过

- 13 项自动测试：P1/NLL 分解的数值和梯度；未知回报掩码与空组；质量锚定和 DPO 参考；等积分 curriculum；SFT 标签掩码；KL 方向；log 分组隔离；不可变文件；unary 不跨候选交互且参数数目相同；full20 拒收缺失分支；配对 log bootstrap；29 列、单位、3 seed 聚合与缺项判定；保存/恢复优化器与 RNG 后参数逐位等于连续训练。
- 真实历史反馈上，新 P1 目标和梯度与旧 `selector_attribution_common.feedback_loss(..., 'P1')` 一致。
- 7 种 residual 结构的真实缓存前向/反向检查。
- B 的全部 13 个目标各完成 4 步 RTX4090 训练：`../runs/20260917_validation_feedback_v3/smoke/`。
- C 的 full、同参数 unary、原始 scorer；F 的三个来源组，均完成 4 步 GPU 训练：`../runs/20260917_validation_c_v1/`、`../runs/20260917_validation_f_v1/`。
- D 用生产 dataset 坐标变换提取 6339 个 token 的专家标签。5 个 token 无有效未来；有效 6334 个，排除清单和新 filter 存在 `../runs/20260917_validation_supervised_v3/supervised_inputs.json`。selector SFT 与同有效样本 GRPO 均完成 4 步 GPU 训练。
- planner L2 与原始 BC 的真实 head 构建、loss 反向和自定义 hook 构建通过。修正了训练入口 `scripts/train.py`、旧 runner 对 `total_epochs` 的假定、custom hook 自动传入 `out_dir` 的兼容性，以及 H100 梯度裁剪的历史兼容问题。
- E 默认历史 64 状态的来源指纹、前缀、场景、策略与排除集审计；生成 1280 个待执行候选分支及 sentinel 清单。物理分支尚未运行。
- common 三套噪声缓存的 oracle 分析已完成，输出在 `../runs/20260917_validation_audit_v1/oracle/`。这是缓存局部奖励诊断，不是正式闭环成绩。
- 真实 B/P1 seed0 已完成 1000 步，并导出 native checkpoint：`../runs/20260917_validation_endpoint_v1/native/pair_nll_seed0/report.json`。963 个冻结张量保持一致；256 个缓存状态上的 logits 最大误差 0，argmax 不一致数 0。该模型尚未经过物理闭环评测。
- 新冻结评测运行时的 planner 与 simulator CPU 导入检查通过；8 种结构的 native head 构建及参数名称/形状匹配通过。
- 原 `20260916_prefeedback_factorial_v1` 的代码与数据冻结检查通过，原评测任务未改动。

部分早期 `validation_*` 目录对应修复前的代码快照，保留作为检查过程记录；不可把它们当正式长实验或在当前代码下直接续训。当前正式队列使用 `20260917_paper_[A-F]_v3` 新 ID（v1 后续启动遇到配额错误，v2 仅作存储验证）。正式启动后不要编辑 `paper_study/`：代码变更会被冻结校验拒绝，应另开版本。

## 仍待目标机器验收

- 四卡 H100 CUDA 扩展探针、planner DDP 的 8 次 microbatch 检查。
- 各新架构在真实图像/模拟器路径上的 native bridge；目前缓存导出一致性不替代此项。
- E sentinel 的实际自然动作复现及全部 1280 个物理闭环分支。
- 六队列的正式 3 seed 结果和完整 29 列评测。
- 新 log-disjoint 数据资产与独立 sealed 测试；各训练来源独立采集反馈后的 P1；最终三次完整 pipeline 重复。这些不是当前固定历史反馈消融已经完成的内容。

队列中任何短训练、桥接、来源或完整性检查失败，都应看对应日志修正后重新验收。旧闭环驱动会保留失败条件的现场；这种情况需先检查并归档该条件，不能直接覆盖失败证据。

机器可读记录及验证日志：`../reports/20260917_implementation_validation/validation.json`。

## 分布式入口修复：`python: command not found`

用户六个任务均在 `run_paper_study.sh` 原第 6 行退出（127）。根因是脚本先用 PATH 中的裸 `python` 读取环境 registry，错误地依赖了已激活的 conda 环境。

修复后先调用共享盘上的 algengine Python 读取 registry，再验证目标解释器可执行，并将其 bin 目录加入子进程 PATH。仍支持显式设置 `STUDY_PYTHON`。不要求任务容器提前激活 conda。

新增 3 项入口回归测试已通过：PATH 中没有 python/python3 时 A–F 的 `plan` 启动全部通过；显式解释器的目录传递给子进程 PATH；无效解释器明确报错。此项只验证启动，不表示四卡 H100 物理验收通过。

修复时检查 `20260917_paper_[A-F]_v1` 六个 run 目录均不存在，所以原命令、原 run ID 可直接重新提交。启动脚本哈希已改变，修复前的本机 validation 快照保持原记录，不改写其冻结身份。

## 手动 H100 调试入口与配额修复后的验证

当前先执行 `bash debug_paper_tasks_4h100.sh`，完整命令保存在 `../RUN_DEBUG_FIRST.txt`。SSH 暂缓，旧 full eval 保留中断现场。Python 路径、HDD2 输出、精简运行时及失败日志处理由六队列共享入口统一修复。

21 项代码测试全部通过，包含一项失败后仍汇总六队列检查的测试。真实历史数据的 E 小规模准备检查完成：8 个状态、候选 0、sentinel 与干预集合复用同一场景文件，输出位于 HDD2；没有在此验证中运行物理闭环。该准备快照保留为过程记录，不用于后续正式运行。

四卡 H100 DDP、闭环模拟器以及六队列联调的通过状态须以目标实例的 `summary.json` 为准；本机代码测试不代表这部分已经通过。正式命令以 `NEXT_PAPER_COMMANDS.txt` 的 v3 为准，前文 v1 入口修复记录是历史记录，不再重提 v1。

## H200 实例首次六任务联调：7b2d74d50d

实际设备为 4×NVIDIA H200。B、C、F 短训练以及 E 的 NR/R sentinel + 候选 0 物理链路通过。A 的 sm_90 扩展检查通过，但 scalar NR bridge 在审计阶段失败；D 的 selector 检查通过，planner L2 在读取图片时失败，BC 尚未检查。完整汇总保存在 `../runs/20260917_paper_debug_7b2d74d50d/summary.json`。

A 的 snapshot record 保存了共享逻辑路径，审计器的 artifact 则使用 HDD2 实际路径。逐条核对记录，实际文件与 SHA256 相同。修复新队列 evaluation.load 返回的运行目录为 resolve 后路径，使提交模拟器与审计的 manifest 路径一致；保留 frozen runtime 路径与旧结果。没有放宽来源审计条件。

D 的配置从 WORLDENGINE_ROOT 推导图片路径，之前继承值为不含 data 的干净代码目录。修复 supervised.worker_env 使用 registry 的 legacy_root 作为数据根，执行代码与导入路径仍用干净代码目录。训练参数不变。新增两项路径回归检查，25 项代码测试全部通过。

重新执行 `bash debug_paper_tasks_4h100.sh` 会使用新代码摘要建立独立诊断目录；上述失败现场和通过记录保留。修复后 A/D 的目标 GPU 检查仍须重新运行验收。

补充真实数据检查：生产 dataset 成功加载 246165 帧并筛选到原定 6334 个 token；随后停止了非必要的全帧图片清点，改为定向检查日志中的四个缺失图片。修正路径后四张 JPEG 均成功解码为 1920×1080，配置的训练 image root 一致，旧/新路径下的轨迹 anchor 字节相同。未据此宣称完整 DataLoader 样本或 GPU DDP 已通过。检查日志：`/tmp/paper_image_path_validation.log`。

## 六任务联调验收通过：0ae2ee9554

验收结果 `PASS_DEBUG_CHECKS`；设备 4×H200，从根 contract 写入到完成约 30.9 分钟。汇总：`../runs/20260917_paper_debug_0ae2ee9554/summary.json`。验收时当前 paper_study 代码与本轮 contract 的哈希完全相同。

- B：13 种反馈目标、C：8 种结构、F：3 个数据来源、D：2 种 selector，均完成 seed0 四步训练，历史数值有限。
- D：planner L2 和原始 BC 均完成四卡 DDP 的 8 次 microbatch，并生成 iter_8 checkpoint；冻结感知参数检查通过。
- A：scalar_v3 与 D4_seed0 的 NR/R 共四项 bridge 均接受，每项 8 场景，无遗漏；scalar 两项历史一致性均通过。accepted 指向的 audit 哈希已复核。scalar 的 rare288 正式开环路径完成全部 288 token。
- E：NR/R 合计 8 个状态的 sentinel 与候选 0 干预分支通过（PASS_TRANSPORT_PROBE_ONLY）。

以上是运行链路验收，不是正式三 seed 实验完成，也不是 E full20 结论。六个正式 v3 run 在验收时尚无训练 manifest、评测 contract 或 attempt status，可按 NEXT_PAPER_COMMANDS.txt 启动。之后仍需每个新模型导出与正式评测时的 bridge 验收。旧 run_h100_full_eval.sh 的中断尚未在此次处理中修复或恢复。
