# P1 双域离线机制分析

目标：解释 P1 与 CL_NLL 在 rare/common 上的差异，再设计双域 PDM 同步改善的方法。
这是研究诊断，不是新方法训练或正式全量评测。原 attribution、decision-feedback、P1 snapshot 均只读。

## 范围与口径

- 输入完整的 attribution run（80 条评测、18 个最终模型、3 个 shared 父节点、原 256 状态）及 P1 seed2 正式快照。
- 输出到新的独立目录；拒绝覆盖已存在输出或与历史 run 重叠的路径。失败产物保留，新尝试用新 ID。
- 默认 CPU 四线程，优化步数为零，没有 rollout/训练/模型导出入口。hash 核验消费的文件，结束时重新核验模型 hash。
- 原始分数逐项复现；rare58/common64、NR/R、3 seeds 独立列出，正式快照只作为独立协议引用，不混合噪声或分母。
- seed 先平均，再按 log 做 10000 次 bootstrap；后验删最大贡献/leave-one-log-out 仅敏感性分析，主结果不删例。
- 保留全部逐 seed NC/DAC/success 分歧，再补每域每模式正负分差最大各三个独立 log；每例分析三个 seed。
- 对 P1/CL_NLL 实际访问状态交叉运行 P1、CL_NLL、Continue；自模型 logits 和 argmax 必须复现历史。分歧后的状态不视为同状态反事实。
- 首违例时刻只使用已有原始 CSV 字段；字段缺失时标 NOT_RECORDED，不从总分推断。

## 损失和梯度

同温度、同 gap、同样本分母下，NLL = Pair + SetMass；因此 P1 = 2 Pair + SetMass，CL_NLL = Pair + SetMass。
三种共同项（dense GRPO、teacher KL、原模型结构）保持不变。此恒等式不是创新声明。

同状态诊断覆盖全部 256 状态、三个 shared 父节点及六组最终模型；报告对内 margin、pair mass、未比较候选最大概率、argmax 和相对 V3 的变化。
训练标签按安全分量改善/双方安全综合分改善/双方不安全/无明确偏好分层。
梯度探针覆盖全部 shared 及最终 P1/CL_NLL/Continue：eval-mode、全部反馈16态批次、每 seed 一个固定 dense 批次，无 optimizer，无参数更新。
报告 Pair/SetMass/NLL/dense/KL 及反馈分层贡献的梯度范数和余弦；探针相关性不等于因果机制。

## 运行

在 `WorldEngine-selector-p1-attribution-v1` 工作树运行：

```bash
PYTHONDONTWRITEBYTECODE=1 /inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python \
  projects/AlgEngine/scripts/diffusiondrive/analyze_selector_dual_domain.py \
  --attribution-run experiments/diffusiondrive/selector_p1_attribution_v1/runs/p1_attribution_20260913_v1 \
  --snapshot-run /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-p1-seed2-snapshot-v1/experiments/diffusiondrive/selector_p1_snapshot_v1/runs/p1_seed2_full_20260913_v1 \
  --output experiments/diffusiondrive/selector_p1_dual_domain_v1/runs/dual_domain_20260914_v1 \
  --threads 4
```

```bash
PYTHONDONTWRITEBYTECODE=1 /inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python \
  -m unittest discover -s research/p1_dual_domain/tests -v
```

输出 `statistics.json`、`training_probes.json`、`decision_cases.json`、`paired_scenes.csv`、`review.md`、`input_manifest.json`。
只有全部阶段完成并重新验证输入后才生成 `report.json` PASS；工程 PASS 不代表方法提升。终点始终 REVIEW_REQUIRED。

## 后续决策

先排除固定权重解释：NLL+λPair，λ=0/.5/1，三 seeds，同500步；再匹配训练缓存上的反馈梯度 RMS，总强度系数仅由训练缓存冻结。
若出现超出调权的有限反馈支持偏差，才设计新目标并在训练侧独立场景验证；不会重启 P3 或用评测分歧案例作训练标签。
若需要完整评价，另行冻结 CL_NLL 全部 seeds，与 P1 在一致原生协议下配对；当前工具没有自动扩展入口。
论文必须与 RAD-2、CRAFT、DriveDPO、CLOVER 等区分具体问题，不能把 Pair+NLL 或闭环 selector 本身当成新颖性。
