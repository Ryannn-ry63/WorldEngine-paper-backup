# 46、15、17、20、27：论文候选图与绘图源码

打开 `index.html` 可按选定顺序浏览所有场景。原编号没有改变：

| 编号 | 场景 |
|---|---|
| 46 | 道路边缘车道保持 |
| 15 | 缓弯道路保持推进 |
| 17 | 左转中减速后继续转入 |
| 20 | 绕绿化岛右转 |
| 27 | 沿车流保持直行推进 |

## 修改图片

解压后进入本目录，运行：

```bash
pip install -r requirements.txt
python plot_cases.py --cases 46 15 17 20 27 --gif
```

单独修改某例后输出到新目录：

```bash
python plot_cases.py --cases 46 --formats png pdf svg --output edited_figures
```

直接编辑 **`plot_cases.py`**：

- `STYLE`：字体、颜色、分辨率、画布尺寸、线宽等。
- `TIMES = [2., 4.]`：截图时刻；支持 0、0.5、…、4 秒原始采样，不插值。
- `STYLE['case_overrides']`：例如 `{'46': {'xlim': [-8, 45], 'ylim': [-10, 14]}}`，同一例所有时刻和模型使用相同视窗。
- `STYLE['show_title'] = False`：关闭图顶端标题。
- `panel()`：面板文字、车身与地图绘制。

数据已经包含在 `data/case_*.json` 中。无需原实验目录、模型权重、nuPlan 或 GPU。默认生成统一 2s/4s PNG、PDF、SVG；加 `--gif` 生成九帧动画。PDF/SVG 可用于论文排版，SVG 文字可编辑。绘图脚本重绘统一 BEV 对照，原先预览图和相机图作为参考原样附上。

## 文件

- `figures/`：统一的 2s/4s 图片与完整 0–4s GIF。
- `original_previews/`：此前已展示的原图；另附 15 原版 GIF。
- `camera/`：原始前视画面对照。
- `data/`：地图、两种模型轨迹、每帧其他物体、时间戳、正式评测分项与来源 SHA256。
- `summary.csv`：NC、DAC、TTC、EP、comfort、PDM、实际行程与累计回退。
- `verification.json`：编号/场景对应关系和原始数据校验。
- `manifest.json`：文件 SHA256，供传输校验。

## 数据与截图注意事项

15、17、27 的 Base 原始采样分别有约 0.321、0.291、0.359 m 的累计车头反方向位移（累计回退，不是首尾净位移）；这是记录本身的低速回退，已原样保留。20、46 两边和本包五例的 SARA 均未检出采样回退。

五例均来自同一 R 评测模式，Base 与 SARA 初始物体和时间戳一致。Base 为 DiffusionDrive 基线；SARA 为在线 rare-log seed0 epoch2 版本。元数据来源与 SHA256 位于各 data JSON 的 `provenance`。

本次只整理所选案例，没有平滑、校正、删除或拼接轨迹。所有帧共用初始自车坐标：x 为初始车头方向，y 向左，单位米。GIF 和轨迹保留 0–4 秒的九个真实记录时刻。图中为实际执行轨迹，不是某一次预测的未来轨迹。每种模型使用其自身闭环中的其他物体位置。

这些是选图材料，不是新增评测结论。四秒之后的行为不在记录中；46 的道路边缘横向差异较细，DAC 改善不应夸大为大幅冲出道路/返回道路。正式 EP=0 不一定代表自车不动，请结合实际行程与 GIF 查看。
