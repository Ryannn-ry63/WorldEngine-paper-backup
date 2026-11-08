SARA 论文闭环案例筛选（2026-11-08）

结论：筛出 5 个可供挑选的真实评测案例；建议优先考虑 1（跟车减速避碰）+2（道路走向/路缘纠偏），或 1+4（安全与路口推进）。

对应论文：../SARA/main.tex 的 NavTrain50 主实验，Base 67.82/65.96 NR/R，SARA online rare-log seed0 epoch2 76.52/78.08。
模型身份：
Base SHA256: c541de3d44479cd824ed3131ff7d2e57249b6657c7ba26973cd35064e9914f87
SARA full-model SHA256: e74bc53e4e332d3d37e378d3c0893cde071f085da8557d75871a56a7961e6bd4
权重来自 research/selector_paper_assets_seed0_20261018/models/{base,online_rarelog_s0_b16_e2}；本次未训练、未重跑或改动原评测。

看图方式：
five_cases_overview.png / .pdf：五组总览。
case_01.png 至 case_05.png：各场景完整闭环轨迹、速度与正式分数。
同名 .gif：同步时间的逐帧 BEV 回放，周围车辆分别使用各自 rollout 的真实状态。
case_XX_camera.jpg：原始渲染前视画面 t=0/2/4 秒，上 Base、下 SARA。
case_XX_trajectory.json：绘图用位姿、速度和时间戳，可重新排版。

轨迹定义与核验：
原始 metadata 共 12 帧，frame_idx 0–2 是历史；frame_idx 3 为共同决策起点 t=0，至 frame_idx 11 为 t=4s。图展示这个完整 4 秒评测片段，不是整段原始录制日志。
轨迹取 ego2global_translation 的实际后轴位置，不是单次预测的 4 秒 plan，也没有把多条预测轨迹拼接。
两模型历史/起始 ego 位姿一致，起点周围车辆 gt_boxes 一致，12 帧时间戳逐项一致。
地图依据 GPKG meta.projectedCoordSystem 从经纬度投影到 nuPlan 世界坐标后，与轨迹一起变换到初始车头坐标。
车辆框使用 nuPlan Pacifica 尺寸：宽 2.297 m，后轴至车头 4.049 m、后轴至车尾 1.127 m。
静态单图灰车显示起点（淡）和末帧（深）；总览只显示末帧灰车。判断碰撞时刻请看 GIF 的同一时刻状态，不比较不同时刻车辆框。

五个候选：

1. 跟车减速，避免碰撞
2021.06.28.20.24.43_veh-38_03385_04952-1786ef7edac65502
Base 在前方有车时继续加速，4s 末速约 8.85 m/s，NC/TTC 失分；SARA 保持跟车并减速至约 0.30 m/s。适合展示纵向安全决策。
R PDM: 0.000 -> 100.000; NC 0->100; DAC 100->100; TTC 0->100; EP 0.000->100.000
NR PDM: 0.000 -> 100.000; NC 0->100; DAC 100->100; TTC 0->100; EP 0.000->100.000

2. 道路走向处纠偏（R 模式）
2021.09.16.19.27.01_veh-45_01749_03230-b02bf210eda55579
Base 的轨迹持续向左侧偏离，在路缘/道路走向变化处 DAC 失分；SARA 较早顺应道路走向。前视图也能看到 Base 接近路缘，SARA 保持在道路上。只作为 reactive 模式案例：NR 下二者都是 100。
R PDM: 0.000 -> 100.000; NC 100->100; DAC 0->100; TTC 100->100; EP 0.000->100.000
NR PDM: 100.000 -> 100.000; NC 100->100; DAC 100->100; TTC 100->100; EP 100.000->100.000

3. 沿弯道继续行驶
2021.09.16.14.39.34_veh-42_00032_00186-cd9d029138325c2c
Base 没有充分跟随道路弯曲方向，DAC 失分并明显减速；SARA 顺路行驶，4s 行驶约 30.51m，Base 约 16.83m。该例 NR/R 同样改善，作为稳健道路合规备选。
R PDM: 0.000 -> 98.179; NC 100->100; DAC 0->100; TTC 100->100; EP 0.000->95.629
NR PDM: 0.000 -> 98.179; NC 100->100; DAC 0->100; TTC 100->100; EP 0.000->95.629

4. 路口右转推进
2021.09.16.13.53.10_veh-42_00860_01069-0802a51b0a1d512c
Base 在路口明显减速，4s 仅行驶约 2.18m；SARA 持续启动右转，行驶约 11.20m，NC/DAC/TTC 均通过。展示右转推进，不能写成完成整次转弯。
R PDM: 62.721 -> 100.000; NC 100->100; DAC 100->100; TTC 100->100; EP 10.530->100.000
NR PDM: 62.443 -> 100.000; NC 100->100; DAC 100->100; TTC 100->100; EP 9.864->100.000

5. 解除低速停滞
2021.08.30.16.16.44_veh-40_00779_01088-551cf39c34f75b25
Base 始终低速，4s 仅行驶约 2.00m；SARA 顺着道路行驶约 22.06m，NC/DAC/TTC 均通过。适合展示低进度恢复，画面不如 4 的转弯直观。
R PDM: 58.333 -> 100.000; NC 100->100; DAC 100->100; TTC 100->100; EP 0.000->100.000
NR PDM: 58.333 -> 100.000; NC 100->100; DAC 100->100; TTC 100->100; EP 0.000->100.000

与 10/6 旧材料的关系：
已找到 research/selector_v3_dac_causal_20261006_r5/analysis/REPORT_ZH.md、case_00–05.png、analyze.py 和更早 campaign。旧图含 baseline/intervention 反事实分支，主要分析旧离线 parent→旧在线 control_s2_e10 的 DAC 退化；它们不是当前 NavTrain50 Base→SARA 的自然闭环对照。
旧 runtime.baseline 是 fully pretrained epoch_100；旧 online selector SHA256 818307e189f58078dde2781d7a85a7fbb88acf9c7a05560e8a0575896b714978。不可直接替换旧图标签。
GitHub 私有备份 WorldEngine-paper-backup 的当前连接没有读取权限；此次以完整本地历史材料为依据，没有声称已核验远端最新版本。

使用边界：
这是按正式结果事后挑选的定性案例，不是新增测试集、均值估计或全局机制因果证明。完整 576 个场景×模式配对结果在 paired_scores.csv，改善和退化都保留。
R: 174/288 提高、68 降低、46 在 1e-5 容差内不变；NR: 170 提高、79 降低、39 不变。
EP 是正式评分器返回的 gated/normalized progress，发生 NC/DAC 失败时的 EP=0 不表示车没有移动；实际行驶距离单独由位姿计算。
两边从同一初始状态开始，后续观测与候选随各自轨迹变化；不能把这组自然闭环图称为每一步固定同一候选集合的因果消融。
案例 2 的 NR 结果无差异；主图、图注必须明确 reactive。
前视图是原仿真器渲染画面，存在渲染伪影；论文建议以实际轨迹/地图为主，前视图作为场景背景。

产物与复现：
selected_cases.json 保存每个场景的模型、评分源、metadata 路径与 SHA256、原始图像路径；paired_summary.json 给全体场景配对汇总。
analyze.py 生成全体配对统计和 12 个初筛图；package_cases.py 生成最终 5 例。
safe_metadata.py 以 pickletools 解析纯数据，不执行 pickle globals/REDUCE 中的函数；已验证数值/字符串/scalar 一致性，拒绝 object dtype 和任意 global。
解释器：/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python
命令：MPLCONFIGDIR=/tmp/sara_mpl_20261108 <解释器> research/sara_qualitative_20261108/analyze.py
然后：MPLCONFIGDIR=/tmp/sara_mpl_20261108 <解释器> research/sara_qualitative_20261108/package_cases.py
本次不改 SARA/main.tex，先供用户挑图。
