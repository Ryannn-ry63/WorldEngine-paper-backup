# 2026-09-17：论文证据与机制探索并行执行

本轮实现位于 `code/paper_study/`，统一入口是 `run_paper_study.sh`。训练、日志、模型、冻结的评测运行时写入 `../runs/<run_id>/`；表格写入 `../reports/<run_id>/`。不新建实验 worktree，不修改正在评测的 D1–D4 实现和结果。新模块目前是内部实验实现，发布前仍需整理依赖和可移植路径。

## 与 0916 主线的关系

维持“固定候选生成器上的 selector 后训练”主线：V3 用 dense GRPO 改善候选选择；闭环反馈补充局部评分没有充分表达的后续影响。P1 是有效的历史候选方案，但其收益、成功率/合规性代价、来源混杂仍需解释，不能仅凭 PDM 提升认定达到顶会要求。MID 是权衡消融，不是已证实更好的最终方法。本轮不继续做 lambda 扫描。

有限反馈包含同一状态下两个实际查询候选。`NLL = Pair + SetMass`，所以 P1 是 `2 Pair + SetMass`，MID 是 `1.5 Pair + SetMass`。这只是代数分解，不是创新声明。需要实验证实：收益来自候选对内部排序，还是把概率集中到被查询的候选集合；未查询候选是否存在更优动作。未知的 18 个回报绝不能当成零回报。

B 队列从原始 V3 rarelog seed0 开始，复用冻结的 common/real rare/synthetic dense replay 和历史反馈，3 个 seed 只改变优化随机性。它不是从各自来源训练 V3、再独立采集反馈的三次完整复现。历史 P1 的先 Pair 后 Pair+NLL 与从第一步开始 Pair+NLL 也不混为同一模型。

## 六条队列

| 队列 | 当前实现 | 解释边界 |
|---|---|---|
| A | 为已有 scalar V3、D1–D4×3 补 open_navtest/open_rare；合并已验收闭环；缓存候选 oracle 与来源审计 | 旧闭环尚未全部结束时，表格明确标记缺失，不填补、不伪造 COMPLETE |
| B | Pair、NLL、P1、MID、SetMass、pair-relative group、two-return group、reference-DPO、local20、去反馈状态 KL、质量锚定、两种次序控制；每组 3 seed | 固定历史反馈下的损失/支持集合机制消融；D2 是已有的 dense+全状态 KL 对照 |
| C | full、feature-only、去几何/route-BEV/scene/set、同参数 unary、原始 scorer；每组 3 seed | corrected common 数据；epoch100 起点；19200 更新，保留 4800 中间 checkpoint |
| D | selector SFT、同有效标签样本的 selector GRPO、planner L2、planner 原始 BC；每组 3 seed | 排除无有效专家未来的样本并记录 token；planner 改生成器和 scorer，不能称为纯 selector 对照 |
| E | 历史 fresh-round 64 状态、20 个候选的真实闭环分支；NR/R，先自然动作 sentinel，再全候选；分析 2/4/8/20 支持集合 | 固定访问状态和固定 continuation 的诊断；不是独立盲测，也不是可部署完整策略上界；标签隔离于训练 |
| F | paired common / paired common+real rare / paired common+real rare+synthetic rollout 的 V3 阶段对照；3 seed，19200 更新 | 固定配对池的来源消融，不是 common 全量；尚未实现各来源独立采集反馈后的 P1 主表 |

B/C/F 的缓存训练每卡一个独立任务，最多并行 4 个；先短训练检查，再正式训练。D 的 selector 部分也按独立任务调度；planner 先执行 8 次 microbatch 检查，再使用四卡 DDP，microbatch 2/card、累积 8 次，即有效 batch 64。planner 的 153600 runner iterations 对应 19200 optimizer updates，中间 checkpoint 在 38400 iterations。

缓存训练与正式评测分开验收：训练结束不等于表格完成。native 导出检查冻结张量和缓存 logits；四卡评测仍需要 CUDA 环境检查、scalar 历史一致性，以及各模型的实际动作桥接。

## 启动命令

在**不同四卡 H100 实例**运行不同队列。每个队列独立 run ID，同一命令可断点续跑。不要把同一 run ID 同时启动两份。

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-paper/code
bash run_paper_study.sh queue --queue A --run 20260917_paper_A_v3
```

其他实例分别运行：

```bash
bash run_paper_study.sh queue --queue B --run 20260917_paper_B_v3
bash run_paper_study.sh queue --queue C --run 20260917_paper_C_v3
bash run_paper_study.sh queue --queue D --run 20260917_paper_D_v3
bash run_paper_study.sh queue --queue E --run 20260917_paper_E_v3
bash run_paper_study.sh queue --queue F --run 20260917_paper_F_v3
```

上面六条应分配给不同实例，不能在同一实例同时争用四卡。只训练可加 `--stage train`；之后用相同命令加 `--stage eval`。A 是补评测，E 是物理分支诊断，不使用 train/eval 分阶段参数。运行中看终端给出的具体日志路径；状态文件的 RUNNING 不是进程存活证明。

查看 B 训练状态和单独汇总评测：

```bash
bash run_paper_study.sh status --run 20260917_paper_B_v3
bash run_paper_study.sh report --run 20260917_paper_B_v3_eval
```

首次提交前，须参照文末验证记录区分本机已验证内容与四卡 H100 尚待验证内容。队列入口自动执行短训练/评测 preflight；任何失败都会停止该队列，不会写成验收通过。

## 表格和统计口径

29 列：Model Name、Note、ADE、FDE；open_navtest 的 6 项；open_rare 的 6 项；CL-NR 的 6 项；CL-R 的 6 项与 Success Rate。ADE/FDE 使用 open_navtest 4 秒、米；常规分数同时输出 0–1 和百分数两个 TSV。Note 写训练数据来源、初始化和暴露属性。完整 3 seed 才计算 mean/sample SD；common64 是额外保持性检查。

`per_scene_metrics.json` 保存场景指标；`paired_log_intervals.json` 按同场景配对、以 log 为 cluster 重采样，给出相对 scalar V3 的区间。此区间条件于固定模型/优化 seed，不表示重新采集数据的不确定性。open-loop oracle 的缓存奖励与正式 NAVSIM 重评分不可跨版本直接比较。

## 尚须执行的论文阶段

这些后续阶段没有被本轮队列自动完成，不能提前对外宣称完成：

1. 在可用资产中审计未暴露的日志，冻结 train/development/sealed 边界。现有 rare288、development58、remainder230、common64 不重新命名成盲测。
2. 用开发集决定是否保留新机制，做预算明确的学习率等调优；不得在历史主表上反复挑选最优配置。当前 B–F 使用固定预注册超参数，不宣称已完成强基线调优。
3. common、rarelog、rarerollout 各自先训 V3，再独立采集来源一致的反馈，验证最终方法；保存访问策略、continuation、查询候选和噪声身份。
4. 最终冻结方法后补三次完整 pipeline 复现，最后开启 sealed 集。优化 3 seed 不能代替此项。
5. 汇总收益与安全/合规代价，确定 P1 是主方法、消融基线，还是由有机制证据的新方法替代。若没有突破证据，不靠更换名称制造新方法。

## 验证记录

验证结果另写 `20260917_IMPLEMENTATION_VALIDATION.md`。本地只有一张 RTX4090；四卡 H100 的 DDP 和完整模拟器运行需要在相应实例实际执行后才能验收。当前文档描述执行协议，不表示六条长实验已经跑完。

## 存储修复后的启动版本

正式重提使用上面的 v3。v1 失败现场保持原样；v2 用于本机 HDD2 存储验证，未作为正式 H100 队列。`registry/paper_storage.json` 配置 HDD2 上的 runs/reports；原 `WorldEngine-paper/runs/<id>` 与 `reports/<id>` 通过符号链接保持可访问。代码目录仍在原处。运行时只复制已登记的约 53 MB 输入，保留 data 链接；不复制旧 core dump、缓存或运行产物。每次启动执行实际写入及 fsync 检查；新任务关闭 core dump，保留正常错误日志。
