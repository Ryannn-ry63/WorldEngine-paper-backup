# 个人存储运行入口（非 data 迁移）

## 现在怎么验证

在用户新开的单卡实例中运行：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-paper/code
bash run_owned_runtime.sh check-gpu --device 0
```

默认检查约数分钟，首次导入及共享盘繁忙时可能更久。无需 SSH、不需要关闭其他实例、不改系统挂载、不启动训练、不读取场景/图像数据；需要所选 GPU 有足够空闲显存。依次验证两环境导入和实际文件归属、基础模型 SHA、MMCV CUDA、gsplat CUDA、一个限定两 CPU/一 GPU 的 Ray worker。任何失败会停止并给出日志。

CPU-only 入口：`bash run_owned_runtime.sh check`。本机已通过 `PASS_NONDATA_CPU`；2026-09-18 用户在 H100 单卡实例完成 GPU/Ray 检查并通过，已核对共享日志；不能据此宣称四卡 DDP 或完整闭环通过。

结果：个人 runtime 目录（实际绝对路径见下）中的 `latest_cpu_check.json` / `latest_gpu_check.json` 和每次 `checks/<timestamp>/`。准确目录：
`/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-owned/runtime`。

## 怎样实现，不是只复制

1. 算法环境原本就在个人 global_user：继续使用 `/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine`，未修改它。曾尝试复制一份，因无需重复而停止；残留仅在个人迁移目录。
2. 模拟器环境恢复到 `WorldEngine-owned/environments/simengine`。离线 conda clone 因缺少安装包缓存失败，未升级任何包；改用此前验证过的原环境归档恢复。文件对比状态在 `simengine_restore_verified.json`，GPU验收独立。
3. 旧 nuplan/navsim/MMCV 源码与 epoch100 已逐项备份校验，全部在 `WorldEngine-owned` 下；现有个人 DiffusionDrive/mmcv/gsplat 和编译扩展保持不动。
4. 用个人目录内的 PRoot 为这个启动进程及其子进程提供旧名称到新文件的映射。工具包从 Ubuntu 官方归档取得，SHA256 与本机 apt 元数据一致；仅解包到个人目录，没有 apt install、宿主 mount、SSH 或系统配置修改。
5. 在该进程视图内，旧 hdd/hdd2 项目私有目录整体隐藏，仅开放已备份的特定映射；公共 data 不在隐藏范围，本阶段仍用已有数据入口。即使验证实例仍在旧项目，也不能借旧私有源码/模型蒙混通过。
6. 历史 artifact 路径名称、原代码与哈希保持不变。日志里看到旧路径可能是兼容名称，`path_map.json` 与预检 JSON 的 backing 字段才表示实际文件位置；检查核对 guest 与 backing 的文件 inode/device 一致。
7. 这是一层迁移期兼容入口，不是安全沙箱，也不是开源代码最终目录设计。PRoot 会增加启动与系统调用开销；此处尚未测量完整训练吞吐，不能承诺零性能损失。新方法研发/最终开源应再把路径显式配置化；不能为此直接改已冻结实验源码。

## 非 data 验收范围

已通过本机 CPU 检查：算法侧 torch/mmcv/mmdet/mmdet3d/nuplan/navsim/模型插件及配置；模拟器侧 torch/ray/hydra/nuplan/navsim/gsplat/环境构造模块；编译扩展加载；Python 子进程；旧项目私有目录隐藏；epoch100 SHA 与冻结值一致。

不表示：data 已迁移、新项目公共数据可见、GPU/Ray 已通过、四卡 DDP 已通过、旧队列断点续跑已验收，或物理闭环指标已一致。

## 原长任务如何继续

原命令将通过 `run_owned_runtime.sh run -- <原命令>` 启动，例如后续 B 的评测：

```bash
bash run_owned_runtime.sh run -- bash run_paper_study.sh queue --queue B --run 20260917_paper_B_v3 --stage eval
```

**这不是现在重提 ABCDEF 的指令。** 入口会先检查：恢复环境文件对比通过、全部个人 runs/reports 已复制且 checksum 校验通过、同一实例的 GPU 小检查通过。结果还在复制时主动拒绝续跑，防止缺失文件造成错误或重新训练。公共 data 可访问性还要单独验收。D 的 planner 原中断阶段没有 checkpoint，不能把重启称作从中断 iteration 恢复。

同一 run 不能在旧入口与新入口两边同时启动。切换前还需要对已复制队列的来源校验与已完成条件复用检查；没有自动改写 registry/paper_storage.json 或原 runs 软链接。进程内映射使它们指向自己的副本，真实旧项目源目录保持原状。

## 是否保存镜像

建议保留当前可用实例镜像，作为系统库、CUDA相关工具和容器环境的恢复依据。挂载的旧项目/global_user 文件不会因为保存容器镜像就自动成为镜像内容；仍需当前迁移。新项目是否能选择同一镜像以平台选项为准。镜像、个人文件备份及目标实例验证缺一不可；不要在迁移验收前回收最后一个可访问源文件的实例。


## 2026-09-18 H100 单卡验收

用户在 `h100-wcj-dsw--433069e398ef-orgvfaeazf` 执行 `check-gpu --device 0`，耗时约 225 秒，得到 `PASS_NONDATA_GPU`。两环境实际模块 backing 都在 global_user；MMCV/gsplat sm90 内核及单 Ray GPU worker 通过。模拟器恢复文件比较亦已通过。证据见 `owned_h100_acceptance_20260918.json`。使用本入口时才具备这些映射；直接运行旧脚本仍可能访问旧项目。当前公共 data 未验收、runs/reports 仍在复制、旧任务续跑和四卡 DDP 未验收。可保存这台已通过验证实例的镜像，并在新项目实例重复检查。
