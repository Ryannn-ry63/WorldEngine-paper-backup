# 在线闭环独立分析

先读 [REPORT.md](REPORT.md)。本目录只读既有 registry、有效训练 report 与模型，不更改奖励、实验登记、模型或正式选模结果。

当前报告包含完整 **32 项接受评测**（24 项 A/B 开发评测及8项基线/间隔诊断）。B_s2 p6 的 R/NR 已补齐并通过验收，完整三种子独立选模判断为 **A/B 均不合格**，与 registry 的 `NO_METHOD_QUALIFIED` 一致。`audit.json` 记录具体读取时间和接受的 attempt。

## 复现开发评测汇总

在仓库根目录执行（不需要 GPU）：

```bash
python research/online_closedloop_analysis_20261001/analyze.py
python research/online_closedloop_analysis_20261001/plans.py
PYTHONDONTWRITEBYTECODE=1 python research/online_closedloop_analysis_20261001/verify_contracts.py --evaluations-only
MPLCONFIGDIR=/tmp/online_analysis_mpl \
/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python \
research/online_closedloop_analysis_20261001/report.py
```

`analyze.py` 重新读取实际 COMPLETE status，核验接受结果和模型链路，计算完整三种子表格及配对统计。`plans.py` 更新轨迹统计。`verify_contracts.py --evaluations-only` 刷新评测配置验收并复用已完成的训练协议审计。`report.py` 重算独立门槛判断与图表。更新后的分析文件替换本目录旧输出，冻结实验不变。

报告已补充 B p6 的置信区间、B−A、p3→p6、安全转移和重复失败解释；A p6 数字及旧 roots48 机制分析保持原结论。B 的NR DAC有所缓解，但R/NR DAC及NR TTC仍未通过原门槛。`selected_cases.json` 的地图案例抽样固定于 A p6，不能称作 B p6 地图复核；B的全部保存计划已纳入计划统计。

## 完整复现

除上述步骤外：

```bash
python research/online_closedloop_analysis_20261001/training.py
python research/online_closedloop_analysis_20261001/diagnostics.py
PYTHONDONTWRITEBYTECODE=1 python research/online_closedloop_analysis_20261001/verify_contracts.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/online_analysis_mpl \
/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-owned/environments/simengine/bin/python \
research/online_closedloop_analysis_20261001/map_cases.py
```

最后再运行 `report.py`。地图重建使用原地图与车身角点规则，仅对保存的第3–10帧实际姿态核验 DAC；不是重新执行完整仿真或完整 PDM 评分。

`training.py` 仅读 committed progress 指向的 report，忽略未提交重试。`verify_contracts.py` 复用原只读候选反馈验收函数，并独立核对场景调度、随机种子、模型版本与历史连续性、B 的冻结采集/顺序更新、评测命令配置。不会重新反序列化全部历史 checkpoint 或模拟分支。

统计约定：所有 CSV 为0–1原始尺度（几何、时间除外）；REPORT 部分结果表为100分尺度。bootstrap 是日志整簇重采样5,000次（固定 RNG seed=0），先平均同场景的三个训练 seed。训练种子标准差单独报告。缺 seed 时不生成完整方法聚合行。
