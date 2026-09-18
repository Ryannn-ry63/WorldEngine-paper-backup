# 恢复说明

## 先判断丢失范围

- 只有实例被收走：global_user通常仍在；使用原共享路径，检查迁移进程/锁和data链接，不必重新复制所有模型或重训。
- 代码丢失但共享模型/环境/结果还在：clone私有仓库main到一个新目录，核对SHA清单，再有选择地恢复 `code/` 和 `registry/`，不要直接覆盖正在运行的工作区。
- 模型、cache、环境、完整结果也丢失：本Git仓库不包含这些大文件，必须从共享存储/独立备份恢复；只有索引和哈希不能重建其内容。不要把Git备份当作全量数据容灾。

## 目录约定

快照刻意保留 `code/` 与同级 `registry/`、`reports/`，对应实际 `WorldEngine-paper/` 结构。`evidence/runs/`是可追溯元数据摘录，不是可直接运行的完整 `runs/`，不要整体改名当作训练输出。

`registry/local_paths.json`和`paper_storage.json`保留冻结历史路径；生产入口通过个人依赖/输出的路径映射保持旧契约身份。不要仅为路径好看批量替换历史JSON——这会破坏哈希和续跑身份。恢复后优先使用 `code/run_owned_tasks.sh`；裸旧入口可能仍指向旧项目。

`WorldEngine-owned`中的simengine环境、依赖源码副本、基础权重、CUDA扩展，及原个人algengine都需实际存在。版本清单见 `registry/environment_versions.json` 与 `registry/dependency_provenance.json`。它们记录版本，不能代替已验证环境本体。系统驱动/系统动态库由实例镜像提供。

## 核验与续跑

1. 用 `registry/backup_inventory.json` 核对快照文件SHA256。`snapshot_sources.json`记录原来源及复制时哈希；部分来源路径可在新环境不可见，属于溯源信息。
2. 恢复完整模型/cache/结果，等待迁移 `PASS_BYTE_COPY_ONLY`；新data链接和所有二级目标必须就位。
3. 按 `code/RESUME_AFTER_MIGRATION.md` 先执行 `audit all`，再分别提交prior/A/B/C/D/E/F。每个正式任务在独立四卡实例执行。
4. 数据仍指向原项目时正式入口默认拒绝；如明确在原项目继续运行，可使用文档中的显式选项。不要绕过迁移锁或改写通过标记。

上一份公开Git备份为P1 seed2：分支 `research/selector-p1-seed2-snapshot-v1`、tag `selector-p1-seed2-v1-20260913`。本次额外保存MID结果、D1–D4及A–F实现与迁移恢复入口。代码历史见 `research/paper-code-20260918`，阶段快照见本仓库main与20260918 tag。

## 本次备份检查的限制

源码快照中 owned task runner 的 8 项测试和 full-eval resume 的 5 项测试通过；paper study 的 25 项测试中，24 项通过，1 项测试的 6 个队列子用例因备份位置改变了目录深度，无法自动找到共享 Python。该启动器的 3 项测试在原 `WorldEngine-paper/code` 目录全部通过。

这是启动器的目录约定：旧入口 `run_paper_study.sh` 在缺少显式解释器时按相对路径寻找环境。新位置恢复时应先指定实际存在的 `STUDY_PYTHON`，并核对 owned runtime 和路径注册表；不能假设任意目录 clone 后即可启动。没有为让备份测试通过而修改冻结实验代码。以上均不代表已完成新的四卡续跑或新 data 验证。
