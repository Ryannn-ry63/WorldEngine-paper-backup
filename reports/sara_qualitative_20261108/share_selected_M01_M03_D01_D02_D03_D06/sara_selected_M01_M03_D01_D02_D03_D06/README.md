# 精选失败案例：M01、M03、D01、D02、D03、D06

仅含指定六例，保留原编号、原图、原始轨迹和对应评测模式。可直接将整个ZIP转发给合作者。

## 六例速览

| 编号 | 模式 | 场景 |
|---|---|---|
| M01 | NR | 侧向车辆尚未通过就继续加速 |
| M03 | NR | 遇到对向来车仍加速通过 |
| D01 | R | 柱廊入口继续推进后越界 |
| D02 | R | 左转后继续加速，压出道路边缘 |
| D03 | R | 路沿附近收速不足，车身越界增加 |
| D06 | R | 减速较晚，多走后贴出路边 |

## 修改图片

解压后进入此目录：

```bash
pip install -r requirements.txt
python plot_cases.py --gif
python plot_cases.py --cases M01 M03 D01 D02 D03 D06 --formats png pdf svg --output edited_figures
```

**直接编辑根目录 `plot_cases.py`**：

- `STYLE`：统一字体、颜色、分辨率、画布大小、线宽等；默认保留原图样式。
- `TIMES=[2.,4.]`：主对比截图时刻，支持0、0.5、…、4秒的原始采样，不插值位置。
- `CASE_OVERRIDES`：按完整编号设置所有面板共用的视窗，例如 `{'D02': {'xlim': [-8, 32], 'ylim': [-8, 20]}}`。
- `M_STYLE`：M组橙色交互车辆等配置；`D_STYLE`：D组配置。

如需修改地图、车身、文字等具体画法，编辑 `M/plot_cases.py` 或 `D/plot_cases.py` 的 `panel()`。它们也可单独运行，例如在M目录执行 `python plot_cases.py --cases 1 3 --gif`。根目录脚本将输出放在 `edited_figures/M` 与 `edited_figures/D`，避免两个组的case_01相互覆盖。

M组另生成2/3/3.5/4秒关键帧，时刻在各JSON的`critical_times`中设置。主截图仍为2/4秒；关键帧文件名保留`critical_2s_4s`后缀，请以面板标题为准。

## 文件

- `index.html`：离线浏览六例图、GIF、诊断曲线、前视画面。
- `M/figures`、`D/figures`：原始PNG、PDF、SVG、GIF和诊断图，均原样复制；PDF/SVG适合论文排版。
- `M/camera`、`D/camera`：原始闭环前视0/2/4秒对照。
- `M/data`、`D/data`：独立绘图所需的全部地图、车身、每帧其他物体、实际轨迹、时间、分项指标及来源哈希。
- `summary.csv`：六例两种模型的正式指标、四秒行程、2/4秒速度。
- `verification.json`：编号与场景token的对应、模式、复核结果和解释。
- `manifest.json`：文件SHA256，供转发校验。

绘图只需要numpy、matplotlib、pillow，无需原实验目录、模型权重、nuPlan数据库或GPU。相机拼图为已有评测渲染，不通过这份BEV脚本重生成。打开index.html即可浏览，无需运行Python。

## 共同来源与读图说明

Base为DiffusionDrive基线（perception50预训练实验）；SARA为V3 online rare-log seed0 epoch2，训练/评测标识`online_rarelog_s0_b16_e2`。M01/M03为NR，D01/D02/D03/D06为R。每张对比图内Base和SARA的模式相同，初始位姿/物体与时间戳匹配；不同模式的交通不应混作同一配置。

上排Base、下排SARA；左2秒、右4秒。固定初始自车坐标和比例；实线为实际执行前缀，点线车身为初始位置，没有箭头或轨迹修正。全部记录为0–4秒、间隔0.5秒，不能外推之后行为或保证采样间安全。

M组橙车按过滤后`anns.track_tokens`跨帧核对身份；M01是正式NC下降且有采样重叠，M03是TTC下降、最近间隙约0.11米、无采样重叠。NC=no-at-fault collision，与“任何接触都没有”含义不同。

D组深色边界与洋红色越界来自辅助地图并集，便于定位问题；正式DAC从原accepted结果读取。辅助几何并非对完整PDM drivable map的逐项重建，不能把辅助面积或深度当成官方DAC定义。D02/D03/D06主要是车身部分越界，不应描述成整车冲出道路。

这些案例支持“更积极推进、但风险或道路余量变差”的描述性讨论，不能单凭片段证明奖励偏置、感知漏检或候选库不足等根因，也不能推出SARA整体比Base更激进。

## 逐例信息

### M01 · 侧向车辆尚未通过就继续加速（NR）

侧向车辆尚未通过时继续加速。Base 等待后起步；SARA 更早加速，2.5–3.5 秒与橙色目标车出现采样车身重叠，正式 NC/TTC 从1降为0。

场景：`2021.09.16.19.27.01_veh-45_01749_03230-10909749099354e6`

[2s/4s图](M/figures/case_01_2s_4s.png) · [诊断曲线](M/figures/case_01_diagnostics.png) · [GIF](M/figures/case_01.gif) · [前视画面](M/camera/case_01_camera.jpg) · [PDF](M/figures/case_01_2s_4s.pdf) · [SVG](M/figures/case_01_2s_4s.svg) · [2/3/3.5/4秒关键帧](M/figures/case_01_critical_2s_4s.png)

### M03 · 遇到对向来车仍加速通过（NR）

面对对向车辆仍加速通过。两边 NC/DAC=1，TTC 从1降为0；SARA 与橙车最近采样车身间隙约0.11米，未检出采样车身重叠。适合讨论风险余量，不写成已碰撞。

场景：`2021.09.16.19.49.00_veh-42_00990_01609-e2ae05340dba502d`

[2s/4s图](M/figures/case_03_2s_4s.png) · [诊断曲线](M/figures/case_03_diagnostics.png) · [GIF](M/figures/case_03.gif) · [前视画面](M/camera/case_03_camera.jpg) · [PDF](M/figures/case_03_2s_4s.pdf) · [SVG](M/figures/case_03_2s_4s.svg) · [2/3/3.5/4秒关键帧](M/figures/case_03_critical_2s_4s.png)

### D01 · 柱廊入口继续推进后越界（R）

原 F05：柱廊入口的弯曲通道。Base 较早收速，SARA 继续推进后越出地图通道，DAC 1→0；辅助最大车身角点超出约2.05米。不能根据该图声称已与柱子发生物理碰撞。

场景：`2021.10.06.07.26.10_veh-52_01245_02064-289a58bd2ab6512b`

[2s/4s图](D/figures/case_01_2s_4s.png) · [诊断曲线](D/figures/case_01_diagnostics.png) · [GIF](D/figures/case_01.gif) · [前视画面](D/camera/case_01_camera.jpg) · [PDF](D/figures/case_01_2s_4s.pdf) · [SVG](D/figures/case_01_2s_4s.svg)

### D02 · 左转后继续加速，压出道路边缘（R）

左转后 Base 减速，SARA 持续加速，随后车身贴出道路边缘，DAC 1→0。辅助最大角点超出约0.20米，属于较细的车身越界，不能夸大成整车冲出道路。

场景：`2021.09.16.14.39.34_veh-42_00032_00186-c7cb2e43b2d053b7`

[2s/4s图](D/figures/case_02_2s_4s.png) · [诊断曲线](D/figures/case_02_diagnostics.png) · [GIF](D/figures/case_02.gif) · [前视画面](D/camera/case_02_camera.jpg) · [PDF](D/figures/case_02_2s_4s.pdf) · [SVG](D/figures/case_02_2s_4s.svg)

### D03 · 路沿附近收速不足，车身越界增加（R）

路沿附近，Base 减速停在前方，SARA 保持更多推进，后段车身越界逐步增加，DAC 1→0；辅助最大角点超出约0.34米。

场景：`2021.09.16.19.27.01_veh-45_01749_03230-0b1250020ea95539`

[2s/4s图](D/figures/case_03_2s_4s.png) · [诊断曲线](D/figures/case_03_diagnostics.png) · [GIF](D/figures/case_03.gif) · [前视画面](D/camera/case_03_camera.jpg) · [PDF](D/figures/case_03_2s_4s.pdf) · [SVG](D/figures/case_03_2s_4s.svg)

### D06 · 减速较晚，多走后贴出路边（R）

简单路段的减速时机差异：SARA 收速较晚，多走之后车身贴出路边，DAC 1→0；辅助最大角点超出约0.23米。画面差异较细，结合局部细节与速度曲线查看。

场景：`2021.08.30.16.16.44_veh-40_00779_01088-63d6fbc938bc58cb`

[2s/4s图](D/figures/case_06_2s_4s.png) · [诊断曲线](D/figures/case_06_diagnostics.png) · [GIF](D/figures/case_06.gif) · [前视画面](D/camera/case_06_camera.jpg) · [PDF](D/figures/case_06_2s_4s.pdf) · [SVG](D/figures/case_06_2s_4s.svg)
