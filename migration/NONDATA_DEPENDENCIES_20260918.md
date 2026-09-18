# 2026-09-18：data 与实验结果之外的迁移依赖

## 核实结论

确实还存在旧项目代码、环境初始化、基础模型及实例镜像依赖。当前 Codex 可以访问 `/inspire/hdd2/project/roboticsystem2/wangcaojun-240208020180` 和 `/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180`；用户新项目实例不可见的情况不代表源文件已删除。这两个 project 前缀都必须排查，不能只改 hdd2。

这是当前入口、关键旧分布式脚本、两个 Python 环境和顶层软链接的审计；尚非所有历史工作树、任意动态导入的完整依赖闭包，也未在新项目实际执行完整任务。

## 已确认依赖与状态

旧源码前缀：`/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj`。
备份前缀：`/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-owned/dependencies/legacy_sources_20260918`。

| 对象 | 当前依赖方式 | 处理与状态 |
|---|---|---|
| `nuplan-devkit`（84 MiB） | 当前 CFPI/paper 环境 PYTHONPATH 指向旧项目；两个环境按论文路径探测均解析到该代码 | 已复制到备份前缀并完成 rsync checksum 校验；尚未切换运行路径 |
| `E2E/navsim_v1`（57 MiB） | 当前环境列表保留这个旧路径；此次 navsim 实际优先解析到个人 `nry/DiffusionDrive/navsim` | 按原版本备份并校验；不要把“路径出现”直接等同于每次实际使用，也不要未经核对就用新版替代 |
| `mmcv`（622 MiB） | global_user 的 algengine 有 editable finder 指向旧项目；plain Python 解析到旧项目，论文 PYTHONPATH 优先解析到个人 `nry/mmcv` | 旧源码已备份校验；个人副本和 sm89/sm90 扩展保留。后续需消除 editable 隐式依赖并独立验收 |
| DiffusionDrive epoch100 基础模型（301 MB） | 旧项目 `WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth`；个人原同名路径不存在 | 已备份到 `WorldEngine-owned/models/diffusiondrive/epoch_100.pth`，SHA256 与冻结记录一致；原 manifest 未改写 |
| `dotfiles/.bashrc` | 旧 `grpo_selector_h100_env.sh`、若干 8hopper/8h100 脚本直接 source 旧项目文件 | 未复制整份私人 shell 配置，也未执行它；已提取相关路径。应以显式独立环境入口代替隐式 source |
| `algengine` Python 环境 | 位于个人 `global_user/.../miniconda3/envs/algengine` | 主体已在个人存储，但“环境在这里”不保证 editable/source 也在这里；已记录 179 个 Python 分发包及 conda 元数据 |
| `simengine` Python 环境 | `/root/miniconda3/envs/simengine`，约 15 GiB，实例本地 | 已记录 252 个分发包及 conda 元数据，已完成 15.194 GB 归档、tar 可读性检查和 SHA256 记录，存于 `WorldEngine-owned/environments/simengine_20260918.tar`；最新状态见 `simengine_archive_status.json`。不是已恢复、已验收的可迁移 conda 环境 |
| CUDA 扩展 | MMCV `_ext.so`、gsplat `csrc.so` 位于个人 WorldEngine/artifacts/toolchains | 当前文件真实位于 global_user；已记录 ELF 依赖。仍依赖匹配的 torch/CUDA/驱动及 H100/H200 实测 |
| CUDA toolkit / 系统库 / 驱动 | 旧 bashrc 指向 `/usr/local/cuda-11.8`（当前实例不存在）；个人 `nry/cuda12.4.0` 存在 | 不盲目套用旧 bashrc。设备驱动属于实例配置，不是复制目录即可迁移的文件 |
| `remote.sh` / `setup.sh` | 历史 SSH 辅助工具在旧项目根目录 | 当前手动运行 paper 入口未发现调用，不能算六队列必需；本次没有执行或修改 SSH 配置 |
| 冻结 runtime / configs / registry | 虽然多数代码在 global_user，配置及运行证据仍记录旧绝对路径 | 复制文件不会自动让续跑成功；须做有来源记录的路径迁移，不直接全局替换冻结 JSON/代码导致哈希漂移 |

三个源码备份均已通过 checksum dry-run；内部三个软链接检查未发现解析到 project 的目标。备份不代表已经将运行切换到新位置。

## shell 初始化带来的隐藏路径

旧 `.bashrc` 除代码库外，还设置旧项目的地图、`E2E/dataset`、`E2E/exp`、`metric_cache_navtest_v1`、WorldEngine 根目录和 CUDA 路径。当前 paper 驱动会覆盖部分 exp/metric-cache 路径，但不应据此假定全部环境变量都被覆盖。数据部分并入 `DATA_REQUIREMENTS_FOR_SENIOR_20260918.md` 核实，不整目录复制旧 E2E 工作区。

当前 Codex 的若干继承环境变量仍是 `/path/to/...` 示例值，因此当前交互 shell 不能作为远端分布式作业的环境证据。此审计通过独立子进程分别检查 plain 与 paper 路径，关闭 GPU bootstrap，只做顶层模块定位；不宣称所有 imports 或 CUDA kernels 均通过。

## 迁移后验收顺序

1. 保全源代码、基础模型、环境归档、个人实验结果和历史来源记录。
2. 确认公共数据在新项目的访问方案；大型公共数据继续暂缓复制。
3. 建立 global_user 下独立环境入口，显式设置 Python/PYTHONPATH、数据、地图、CUDA 扩展，去掉对旧 `.bashrc` 的依赖。
4. 在新项目实例检查实际模块来源、文件 SHA、动态库和 H100/H200 内核；再通过已有 bridge 和动作一致性门槛。
5. 验收后才切换续跑路径与新输出存储；冻结历史结果保留原始来源，不用批量字符串替换伪装原实验。

## 证据文件

- `nondata_runtime_audit_20260918.json`：两环境模块来源、包版本、conda 元数据、依赖仓库 HEAD。
- `runtime_source_backups_20260918.json` 与 `backup_source_*_verify.log`：源码复制与 checksum 校验。
- `source_backup_symlinks_20260918.json`：备份树软链接审计。
- `epoch100_backup.json`：基础模型 SHA256 备份证据。
- `cuda_dependency_locations_20260918.json`：扩展真实位置与 ELF NEEDED/RPATH。
- `simengine_archive_status.json`：实例环境归档实时状态，PASS_ARCHIVE_CREATED_NOT_RESTORED 仅表示归档完成，不表示恢复运行通过。
- `top_level_symlinks_20260918.json`：工作区顶层链接，未覆盖所有深层链接。

