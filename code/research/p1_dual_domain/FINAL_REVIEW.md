# 双域分析最终复核与运行说明

本说明补充 README：基础分析的 PASS 表示特征和缓存核验完成；**必须再通过真实物理状态核验，才使用最终决策对照结论**。

本次已执行：

- 基础 run：`experiments/diffusiondrive/selector_p1_dual_domain_v1/runs/dual_domain_20260914_v1`
- 最终 run：`experiments/diffusiondrive/selector_p1_dual_domain_v1/runs/dual_domain_20260914_verified_v1`

最终 `report.json` 引用基础统计、训练探针和物理状态校验后的 `decision_cases.json`、`interpretation.json`、`review.md`。
原始 run 不覆盖、不重写；模型参数更新0次，新闭环0次。当前与历史模型均未替换。

## 真实状态核验

```bash
PYTHONDONTWRITEBYTECODE=1 /inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python \
  projects/AlgEngine/scripts/diffusiondrive/finalize_selector_dual_domain.py \
  --analysis-run experiments/diffusiondrive/selector_p1_dual_domain_v1/runs/dual_domain_20260914_v1 \
  --output experiments/diffusiondrive/selector_p1_dual_domain_v1/runs/dual_domain_20260914_verified_v1
```

已存在的输出会被拒绝，复跑用新的输出目录。若首次基础分析尚未完成，先运行 README 中的基础分析命令。
核验读取实际 pre-action ego 状态、周围交通位置/速度、时间、交通灯，再与 selector 可见输入、候选和执行前缀同时匹配。
这些特权状态只用于分析，绝不进入 selector。分歧后即使特征重新接近也不重新认作同前缀因果对照。
原始 CSV 缺失首违例时间时保留未知，不根据最终分数捏造时间。

## 下一轮最小对照（未启动）

共同主目标是 rare/common PDM 同步改善，NR/R 分列；安全分量、进度、rescued/broken 全部报告。
先排除固定权重解释：NLL+λPair，λ=0/.5/1，三个 seed，同 shared 父节点、optimizer/RNG、256状态、dense cache、500步、原20候选。
再加 MID_SCALE_MATCH：λ=.5，训练开始前在相同父节点的完整训练缓存16态批次上计算
`sqrt(sum ||g_NLL+g_Pair||² / sum ||g_NLL+0.5*g_Pair||²)`，每 seed 冻结一个系数，仅缩放整体反馈项，不缩放 GRPO/KL。
分母为零时停止，不根据开发集选择系数。只评估最终step，端点对照只有输入/噪声/模型/协议一致时复用。

固定调权若已解决大部分差异，应作为工程控制而非新方法贡献。只有有限反馈支持偏差在训练侧独立场景得到验证，才设计新目标。
未知候选缺少真实反事实回报，当前交叉分数和梯度不能证明其导致 common 碰撞。不得自动重启P3，也不得用本次后验评测案例作训练标签。
完整 CL_NLL 三 seed 原生评测另行决定，不挑 observed-best seed，不和开发噪声混算。
