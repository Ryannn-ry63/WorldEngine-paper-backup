# 2026-09-16：论文路线与首批归因实验

## 当前决定

P1 有完整闭环 PDM 的均值收益，但尚未定稿。MID 未形成稳定折中，暂停继续扫 Pair 权重。目标仍是奖励驱动的 selector 后训练；先回答收益和退化来自哪里，再决定是否需要新方法。时间和成本不作为主要限制。

现有模型均为固定 V3 rarelog seed0 起点上的 mixed replay + 闭环反馈续训。三优化 seed 不是三次独立完整流程。新工作区不改变或删除原实验产物；最终公开代码还需进一步裁剪依赖、移除历史原型。

## 执行顺序与交付

1. 建立独立源码快照及来源哈希；保存已有结果的 29 列、均值/标准差和来源说明。
2. 从原始、未接受闭环训练的 V3 开始 D1–D4，分离 dense 来源变化与偏好反馈。
3. 在同反馈、同预算下比较局部完整候选监督、闭环组内相对奖励优化、Pair、NLL、Pair+NLL。未知候选不补造回报；另控制已查询集合的总概率质量。
4. 固定状态、候选、噪声和续接策略，对独立开发日志做完整候选分支诊断，检验排序错误、反馈覆盖和 DAC 失败。后验案例与梯度相关性不作为因果证明。
5. 只有机制证据支持时改进方法；冻结之后在 common/rarelog/rarerollout 分别做来源一致的主表和完整流程重复。历史 rarelog→mixed 独立列出。

本次实现只覆盖第1项和第2项的训练准备、训练及缓存验证；闭环评测和后续实验必须有独立执行与验收记录，不能把训练 PASS 当作论文效果 PASS。

## D1–D4 冻结契约

| 组别 | hard replay | 闭环偏好 |
|---|---|---|
| D1 | 真实 rare | 无 |
| D2 | 真实 rare + 合成 rollout | 无 |
| D3 | 真实 rare | 原固定 F，Pair+NLL |
| D4 | 真实 rare + 合成 rollout | 同一 F，Pair+NLL |

- 四组从原始 rare_tuned seed0 selector 开始，SHA256 `aed5e36f1ccb0df96b703a901b54f48c41dcf1967cc86d63519eb4da37519eb2`；不加载 shared/P1 optimizer。
- 每组三个优化 seed，各 1,000 步；AdamW lr/weight decay=1e-4、clip=10、T=1、KL(V3||current)=0.01。
- 每批 common8+hard8+feedback16（NR8/R8）；所有组对同一完整32态批次做 teacher KL。D1/D2 不使用反馈偏好标签，但会访问反馈状态用于 KL，所以不能称为完全不接触闭环状态。
- common 抽样基于同一个旧 paired-common 映射池，在四组固定；hard 分别从 real 子集或原 mixed 记录池抽样。共同项与反馈用独立 RNG 流预生成，保存全部 draw。
- common 池包含旧 mixed 记录所配对的 common 样本，因此 D1/D3 应称为“固定 common 池、无合成 hard replay”，不能在成员审计前视为完全复刻原 rarelog 训练源。
- 反馈 F 为原256态/每态两候选的离线反馈，包含 V3 和 shared 访问/续接策略。不是新模型的 on-policy 回报。保留原 gap、Pareto 标签与全反馈状态分母。
- D3/D4 从第1步使用 Pair+NLL，没有历史 P1 的先500步 Pair 再500步 Pair+NLL 调度。受控 D4 不冒充已发布 P1。
- dense 与 feedback 的来源及日志交叉写入 `data_audit.json`；额外状态/KL、同反馈优化方式和阶段调度仍需独立对照。

## 入口和输出

在本目录的上一级 `code/` 运行：

```bash
bash run_study.sh export
bash run_study.sh prepare
bash run_study.sh validate --device cpu
bash run_study.sh validate --device cuda
bash run_study.sh smoke --device cuda
bash run_study.sh train --arm D1 --seed 0 --device cuda
bash run_study.sh status
```

训练对 D1–D4 和 seed0/1/2 分别执行。不同组可使用不同 GPU 并行；同一组同一种子使用锁避免重复写入。训练每25步保存 optimizer/RNG；改变代码/契约必须创建新 run ID。smoke 使用单独目录，不能当作正式模型。

`registry/local_paths.json` 保存本机路径；源码不复制模型/数据。输出统一在工作区 `runs/`、`reports/`，历史输入只读引用。独立快照提交、源码哈希和旧未提交补丁位于 `registry/migration.json` 与 `archive_index/`。

## 论文验证要求

按 log 审计训练/开发/验证交叉。common64、development58、remainder230 均不作为新盲测。方法选择只使用开发数据；没有合格独立集合时增加新场景，不改名旧集合。

正式主表保留29列：模型、Note、ADE/FDE、OpenLoop-navtest、OpenLoop-rare、NR、R各六指标、R Success=NC&DAC。ADE/FDE 为4秒米制，其余0–1和百分制分别导出。另报 NR success 与 common64 retention。

后续主表各数据组统一阶段A预算，覆盖19,200步并保存4,800步检查点；阶段B统一预算。三个完整流程 seed 各自训练 V3、采集反馈、续训；评测噪声重复不算独立训练。纳入可比强基线、监督后训练、V2/V3结构消融。

效果验收同时关注 NR/R PDM、Success/DAC 和完整开环保持。0.5个百分点非劣界限是研究阶段预设标准，不是录用门槛。置信区间跨零不能证明保持；样本不足时结论为证据不足。

## 研究主线

“更好的轨迹已经存在”需要候选覆盖与 oracle regret 证据；V3 的结构作用需要公平消融；P1 的闭环反馈作用需要排除额外数据和训练量。GRPO 并非不能使用闭环奖励。若更简单的同反馈 GRPO/NLL 足够好，保留更简单方法。

与 RAD-2、CRAFT、DriveDPO、CLOVER 区分具体监督、支持范围和执行机制，不能把纯selector、Pair+NLL或一般闭环重排序当成新颖性。相关文献：

- https://arxiv.org/abs/2604.15308
- https://arxiv.org/abs/2605.04470
- https://arxiv.org/abs/2509.17940
- https://arxiv.org/abs/2605.15120


## 2026-09-16 执行结果与原生导出入口

D1–D4 × seed0/1/2 共12个正式模型已训练完成，每个1000步。CPU/GPU的256态一致性验证、四组短训练和模型/optimizer逐项一致的断点恢复均通过。占卡监督脚本已恢复原tmux中的程序，并独立确认GPU显存占用恢复。动态状态及证据以run/registry记录为准。

原生导出只替换54个selector张量，序列化后核对另963个张量不变，并对256态核对logits/argmax。此检查不是物理仿真bridge。

在 `code/` 执行：

```bash
# 全12个模型的CPU原生导出；也可指定 --arm D1 --seed 0
bash run_native_export.sh --all
# 查询训练进度，无需GPU
bash run_study.sh status
```

`scripts/audit_study.py` 使用 `registry/local_paths.json` 中的 `training_python` 运行，核验12份正式训练报告、selector/恢复文件/原生checkpoint哈希及导出报告，产出 `runs/20260916_prefeedback_factorial_v1/acceptance_training.json`。`PASS_TRAINING_AND_NATIVE_CACHE` 仅验收训练及缓存推理，不代表闭环评测完成。

GPU使用监督脚本为 `scripts/with_gpu_holder.py`。PID必须来自当次进程检查，不可沿用旧PID；只释放验证过的 `python occupy_GPU.py`，异常退出也尝试恢复。实际恢复需用tmux和nvidia-smi独立确认。

下一阶段：迁移后的物理仿真衔接验证、D1–D4闭环归因评测、开环/common保持检查。现有旧评测入口的方法白名单不包含D1–D4，不能冒用P1/MID名称直接运行。新增评测实现需单独冻结代码和协议，保留当前训练代码哈希；不能修改训练契约后继续往同一run写训练结果。
