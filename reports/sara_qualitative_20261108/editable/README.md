# 可编辑的 SARA 论文绘图包（案例 01–10）

图是用 **Python + Matplotlib** 绘制的。这个目录包含独立绘图脚本、样式配置、十个案例的地图与闭环轨迹数据，以及 PDF / SVG / PNG 输出。在自己电脑上重画只需 NumPy 和 Matplotlib，不需要训练 checkpoint、原始 nuPlan 地图、仿真器或服务器路径。

[下载完整可编辑包](sara_editable_figures.zip) · [返回案例浏览](../README.md)

## 运行

下载并解压完整包，进入目录后：

```bash
python -m pip install -r requirements.txt
python plot_cases.py --cases 6 7
```

默认输出到 `figures/`，每个案例同时生成 PDF、SVG、PNG。省略 `--cases` 会重画全部十例。

```bash
python plot_cases.py --cases 6 7 8 --formats pdf svg --output paper_figures
```

`data/` 是从已核验的真实闭环记录导出的数据，不是从 PNG 反向提取或手工重画的轨迹。包含每 0.5 秒的位姿、速度、车辆轮廓、地图形状、R/NR 指标和来源哈希；绘图统一用 R 模式。JSON 可直接阅读。

## 最常改的参数

直接编辑 `style.json` 后重新运行，不需要改数据：

| 参数 | 用途 |
|---|---|
| `colors` | Base / SARA 的配色 |
| `model_labels` | 图中的方法名称，如将 Base 改成 DiffusionDrive |
| `font_family`、`font_size` | 字体与字号；需要本机安装对应字体 |
| `figsize_with_speed`、`figsize_bev` | 图宽、高，单位英寸 |
| `show_speed` | 是否保留右侧速度曲线；设为 `false` 只画两张 BEV |
| `show_metrics` | 速度图内的 NC / DAC / TTC / EP 指标框 |
| `show_scores`、`show_title`、`show_footer` | 得分、总标题、页脚说明的开关 |
| `trajectory_linewidth`、`marker_size` | 轨迹线宽与采样点大小 |
| `show_footprints`、`footprint_stride` | 是否画 ego 车身轮廓，间隔几个采样点 |
| `actor_frames` | 周围车辆取哪些帧；`[0,8]` 是 t=0 与 t=4 秒，`[8]` 只保留末帧 |
| `hidden_map_layers` | 隐藏指定背景层；例如 `["baseline_paths"]` 隐藏道路中心参考线 |
| `case_overrides` | 针对单个案例调整标题和裁剪范围 |
| `dpi` | PNG 分辨率；PDF / SVG 中轨迹、地图和文字为矢量 |

例如把 `case_overrides` 改成：

```json
{
  "6": {
    "title": "Right-turn road compliance",
    "xlim": [-7, 24],
    "ylim": [-12, 10]
  }
}
```

Base / SARA 使用相同裁剪范围和等比例坐标。地图之外的任意标注、箭头或不同的面板布局，可以改 `plot_cases.py` 的 `draw_case()`；地图外观改 `draw_map()`。`footprint()` 使用原车辆尺寸和后轴偏置，不建议为视觉效果修改几何尺寸。

## 论文和手工排版

- LaTeX 建议使用 `figures/case_06.pdf` 等矢量文件；放大不会像 PNG 一样模糊。
- 用 Inkscape / Illustrator 做手工排版可打开 SVG。文字保留为文字对象（`svg.fonttype=none`），路径也是独立矢量对象；在其他电脑打开时仍需要相应字体。
- PDF 使用 TrueType 字体嵌入（`pdf.fonttype=42`）。PNG 是方便预览的版本。
- 这套脚本重新生成静态图；原来的 GIF 和仿真前视画面保留在第一批 / 第二批目录。前视画面本身是栅格图，不能变成可编辑的道路矢量。

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/case_06.pdf}
  \caption{Closed-loop trajectory comparison over a 4-second reactive rollout.}
  \label{fig:sara_case06}
\end{figure}
```

## 保留结果含义

每例覆盖完整四秒评测片段，而非整段驾驶日志；0.5 秒一个采样点。灰色车辆默认表示起点和末帧，不表示所有中间位置。两边初始状态一致，后续 reactive 交通分别演化。调整画面不应改变轨迹坐标、速度和评测数值。

案例②仅在 R 模式体现优势；⑨有推进效率下降的代价；⑩只展示越界前收停，未证明之后能恢复正常行驶。论文图注应保留相关限定。优先推荐⑥、⑦，交互安全可选⑧。

`export_inputs.py` 是服务器端导出记录，依赖原研究目录的 `analyze.py` 及完整数据。**本地使用只运行 `plot_cases.py` 即可**；无需运行导出脚本。
