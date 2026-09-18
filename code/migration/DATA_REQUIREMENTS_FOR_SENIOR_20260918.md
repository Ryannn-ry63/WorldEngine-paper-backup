# 2026-09-18 论文数据依赖与迁移范围（供与学长核实）

结论：需要保留“当前任务 + 完整论文后续”的数据访问能力，但不需要直接复制整个 WorldEngine/data 链接树。当前已知新项目不能访问旧 roboticsystem2 路径；旧项目里的 Codex 仍能读取。是否存在新项目可用的同版本公共副本或合规共享挂载，需要向学长/平台核实，不能凭当前 Codex 可读判定新项目可读。

大型公共数据仅盘点，未启动复制。个人实验 runs/reports 继续向 global_user 复制。所有迁移目的地仍限定为 nry 下的 WorldEngine-owned。

## 1. 当前六队列实际需要什么

- A（已有模型开环/表格）及 B/C/D/F 的原生评测：open_navtest 为 12,146 tokens，open_rare 为 288 tokens；需要对应原始图像、navtest 标注、评分缓存、模型与评测代码。闭环需要 rare288 + common64 的场景及数字孪生资产；bridge8 被当前 rare 集资产覆盖，不应额外加一次容量。
- B/C/F 的 selector 训练主要复用已有候选/奖励/特征缓存；这些缓存训练本身不要求重新读完整原始图像库，但正式评测仍需要数据。不能以“缓存训练能跑”为迁移验收。
- D 的 planner L2/BC 训练确实需要原始 navtrain 图像和标注：当前有效专家集合 6,334 tokens；不是仅有 selector 缓存就足够。
- E 的历史 full20 诊断当前覆盖 50 个场景，实际映射到 48 个背景资产；来自 128 场景反馈集合。场景 ID 与资产 ID 并不总相同，已按 metadata.digitaltwin_asset_id 和实际 renderer 加载逻辑核对，不能直接拼场景名复制。
- 其余个人模型、dense replay、反馈快照、合成 rollout、manifest/audit/source provenance 继续保护；这些不能因已存在公共原始数据而删除。

当前精确资产容量（GB 为十进制，1 GB = 10^9 bytes）：

| 集合 | 场景数 | 去重资产数（组内） | 背景模型容量 |
|---|---:|---:|---:|
| bridge | 8 | 8 | 2.90 GB |
| common | 64 | 64 | 24.59 GB |
| full | 288 | 288 | 105.59 GB |
| E_current_targets | 50 | 48 | 18.30 GB |
| feedback_full_cohort | 128 | 117 | 44.94 GB |

当前三组评测 + E 目标合并去重，共 **400 个背景模型，148.47 GB**。将完整反馈 128 场景也纳入后共 **175.12 GB**。各行不能直接相加，有重叠。当前精确资产文件检查缺失数为 0。以上只计 renderer 实际读取的背景 checkpoint，**不是训练/评测全部数据总量，也不是完整运行验收**。

完整逐文件清单：`paper_data_requirements_20260918.json`；生成脚本：`audit_paper_data_requirements.py`。场景 pickle 本身约：rare288 0.912 GB、common64 0.254 GB、feedback128 0.524 GB（后两项已在个人 global_user 历史实验中）。

## 2. 后续论文需求，不能缩成眼前子集

依据 0916 执行计划、0917 并行实验计划及 0827 meeting note，仍需：

1. common / rarelog / rarerollout 来源对照，以及各来源先训 V3、独立采集反馈后再训练最终方法。
2. 三次完整 pipeline 的复现，不能只重复优化 seed；需要能重新计算候选、重采集 rollout 和反馈。
3. 对未暴露日志建立 train/development/sealed 划分，冻结方法后做独立验证。当前 rare288、common64 和历史诊断集合不能替代新 sealed 集。
4. 监督续训、原 selector/结构消融、候选 oracle、完整开环和 NR/R 评测。

所以未来必须保留原始 navtrain/navtest 的相关图像、标注/元数据、地图、训练场景和相应可用数字孪生资产。旧 V3 数据协议在全 navtrain 103,288 tokens / 1,192 logs 上挖掘 rare，得到 6,271 rare tokens；如果从头重做这一步或做完整 common 训练，不能仅保留 D 的 6,334-token 子集。

未来具体训练/开发/sealed 集尚未冻结，因此还不能给出完整论文所需的最小图像文件清单及最终容量。新场景需要的资产集合也尚未冻结。可优先核实全 navtrain/navtest 公共库的持续访问方案，再按选定日志和真实文件引用制作去重迁移清单。不能声称当前约 175 GB 资产就覆盖以后所有实验。

common / rarelog / rarerollout 是训练数据来源轴；open_navtest / open_rare 及 CL-NR / CL-R 是评测轴。它们不是三个可以仅按文件夹名分别搬运的数据集。

## 3. 给学长核实的物理数据目录

下列路径是实际物理来源，不是个人目录里看起来存在的 data 软链接。

| 物理目录/文件 | 用途与必要性 | 容量/处置建议 |
|---|---|---|
| `/inspire/hdd2/project/roboticsystem2/public/nuplan/dataset/nuplan-v1.1/sensor_blobs` | 原始相机图像；当前开环、D planner 训练，以及未来重新提取特征/全流程实验需要 | 整库 du 约 17 TiB（传感器根目录，不等于论文必需图像子集）；先核实能否继续访问或已有同版本副本，再按实际 token 的相机/历史帧选取。不能只取目标 token 的单帧 |
| `/inspire/hdd2/project/roboticsystem2/public/worldengine/data/WE_release_data/alg_engine/merged_infos_navformer` | 当前 navtrain/navtest 标注 | 两份当前必需 pkl 合计 16.97 GB；目录 du 为 36 GiB，另含约 20.70 GB goalflow 标注，当前方法未需要这份额外标注 |
| `/inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1/meta_datas` | 场景重建、日志索引、未来重建缓存与新数据划分 | 约 15 GiB；navsim_logs 指向它；应与标注和图像版本保持一致 |
| `/inspire/hdd2/project/roboticsystem2/public/nuplan/dataset/maps` | 地图与评分/场景重建相关依赖 | 约 1.4 GiB；已有评分缓存不代表未来不需地图 |
| `/inspire/hdd2/project/roboticsystem2/public/worldengine/data/nuplan/WE_processed/navtest_failures/assets` | rare288 / bridge 闭环渲染 | 当前 288 个背景模型合计 105.59 GB；源目录早先 du 约 100 GiB（单位不同，勿相加） |
| `/inspire/hdd2/project/roboticsystem2/public/worldengine/data/nuplan/WE_processed/navtrain/assets` | common 保持性、E、训练反馈以及未来 rollout/新集合 | 全量约 4.7 TiB；当前只需部分资产，未来范围随正式集合确定，不能先认定全量无用 |
| `/inspire/hdd2/project/roboticsystem2/public/worldengine/data/original_scenarios_WE` | 原始场景；当前 rare288 使用 navtest_failures/combined/all_scenarios.pkl | 当前文件约 0.912 GB；整个父目录约 92 GiB，不等于当前必需量 |
| `/inspire/hdd2/project/roboticsystem2/public/worldengine/data/WE_release_data/sim_engine/scenarios` | 既有训练/测试/增强场景，未来场景生成与对照可能用到 | original/navtrain 约 398 GiB；original/navtest 约 6.7 MiB；augmented 约 539 MiB。需对照具体实验选取，暂不整目录复制 |
| `/inspire/hdd2/project/roboticsystem2/public/worldengine/ckpts` | 公共预训练模型/其他方法基线 | 约 16 GiB；当前 DiffusionDrive 的 epoch100 模型另在个人旧项目目录，不在这个公共模型集合里。只迁移实际模型及计划使用的对照模型即可 |
| `/inspire/hdd2/project/roboticsystem2/public/worldengine/algengine_cache` | 公共其他方法缓存 | 当前已有个人 NAVSIM metric cache 和 V3 候选缓存；不能仅因 data 链接而全搬。若重跑学长原方法基线，需重新核对具体依赖 |
| `/inspire/hdd2/project/roboticsystem2/public/worldengine/data/openscene-synthetic` | 公共合成数据/学长其他模型训练数据 | 我们的 V3 rollout 使用自己采集的合成记录；不是整个公共 synthetic 目录都必需。若重跑原方法，对应子集再纳入 |

注：`data/raw/openscene-v1.1/sensor_blobs/trainval`、`test`、`mini` 目前都链接到同一个物理 sensor_blobs 根目录，不能复制三次。同样不能把 `raw/nuplan` 和 OpenScene 图像树当作两套独立原始图像再各复制一遍。迁移全链时还需检查嵌套软链接。

## 4. 已有个人资产与还要救出的模型

- `nry/exp/metric_cache_navtest`、`metric_cache_trainval`、V3/P1/MID 及后续候选缓存/历史实验保持不动；旧 stage 清理没有删除这些缓存。
- `WorldEngine-paper/runs/reports` 中部分实际位于旧项目个人目录，正在复制到 `WorldEngine-owned/project_outputs/WorldEngine-paper`，尚未整体验收。
- epoch100 基础模型原物理路径：`/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth`，约 301 MB。个人 global_user 原同名路径不存在，已单独备份到 `WorldEngine-owned/models/diffusiondrive/epoch_100.pth`，并通过既有 SHA256 校验；见 `epoch100_backup.json`。旧模型路径及实验契约尚未切换。
- 新项目还需核对 nuplan/navsim Python 代码依赖、simengine 环境与 CUDA 扩展；这些不是数据集，但同样影响切换后运行。

## 5. 建议直接发给学长的文字

> 学长，我核对了一下：代码主要在 global_user，但 WorldEngine/data 是链接到 roboticsystem2 公共数据；换项目后原路径在新实例里目前访问不到。我们当前和后续论文需要 navtrain/navtest 的原始图像、标注/元数据、地图，以及 WE 的场景与数字孪生资产。当前闭环用到的背景模型按实际引用去重约 148.5 GB，保留完整历史反馈集合约 175.1 GB；但后续独立反馈采集、common/rarelog/rollout 全流程重复和新测试集合，还会用其他训练资产。全量 navtrain 仿真资产约 4.7 TiB，加上原始图像，个人目前约 2.2 TiB 空间放不下。想确认：换项目后这些公共数据是否有允许继续使用的共享挂载或同版本公共副本？如果必须全部放 global_user，我们再按正式实验所需日志/场景列子集迁移，而不直接复制整个 data。自己的实验结果我会搬回 global_user。

## 证据与边界

- 当前 run 输入：`runs/20260917_paper_B_v3_eval/inputs.json`、`paper_D_v3/supervised_inputs.json`、`paper_E_v3/mechanism.json`。
- 计划依据：`docs/20260916_EXECUTION.md`、`docs/20260917_PARALLEL_EXPERIMENTS.md`、旧 `0827_meeting_note.md` 与 V3 rare-original/rollout runbook。
- 资产加载逻辑：`projects/SimEngine/worldengine/render/mtgs/mtgs.py` 中 `digitaltwin_asset_id`、`AssetManager.reset/load_asset`。
- 容量依据：`public_data_small_sizes.log`、`paper_data_requirements_20260918.json` 和前期 du 记录。sensor_blobs 全库盘点已完成，du 约 17 TiB；该值不代表论文所需子集容量。整库加 navtrain 仿真资产已超过个人 10 TiB 总配额，不能整体迁入。
- 新项目不可见以用户此前 mount check 的结果为依据；本次未远程登录新实例，也未确认存在可替代副本。

## 最新决定：2026-09-18 新项目存储

用户确认学长允许将公共数据迁入当前新项目，再通过个人工作区软链接使用；这取代此前公共数据必须全部放 global_user 的限制。自己的实验结果和非data依赖仍按现方案保存在 global_user。新项目目的地绝对路径及由谁执行公共数据复制待确认。容量扫描已按用户要求停止，保留未完成统计；没有启动公共数据复制，也没有切换现有 data 链接。目标数据复制并校验后再切换。
