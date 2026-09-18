# 2026-09-18：GitHub 备份、同步上游与向学长提交 PR 操作指南

以后可以直接对 Codex 说：**“请看 0918 的 GitHub / 上游 PR 指南，按这份指南整理最新代码。”**

本文件名：`20260918_GITHUB_UPSTREAM_PR_GUIDE.md`。本地入口在 `nry/WorldEngine/`；GitHub 副本在私有备份仓库的 `docs/`。这是后续执行指南；2026-09-18 本次只更新私有备份，没有向公开 fork 推送，也没有向学长提交 PR。

## 1. 用户的长期要求

- 保留学长 WorldEngine 的原有框架，新增方法尽量放在独立模块中；确有必要修改框架时，单独说明原因、兼容性与验证结果。
- 可以同步学长后续更新，也可以把整理好的贡献通过原 fork 向学长提交 PR。
- 论文开源版要去掉废弃探索、临时脚本、个人路径和内部材料；私有研究备份可以保留这些追溯信息。
- 活跃实验和冻结 runtime 不直接切分支、合并上游或批量改路径。整理工作在独立 checkout 中进行。

## 2. 三个仓库及两类分支

| 名称 | 地址 / 分支 | 用途 |
| --- | --- | --- |
| 学长上游 `upstream` | https://github.com/OpenDriveLab/WorldEngine | 同步框架、接收最终 PR |
| 原公开 fork `fork` | https://github.com/Ryannn-ry63/WorldEngine | 推送经整理的公开代码分支，再向上游提 PR |
| 私有备份 `origin` | https://github.com/Ryannn-ry63/WorldEngine-paper-backup | 保存尚未发表的代码、方案、结果和恢复资料 |
| 私有备份 `main` | `code/`、`registry/`、`reports/` 等工作区结构 | 阶段快照，不直接作为上游 PR 分支 |
| 私有代码历史分支 | `research/paper-code-20260918` | 原 WorldEngine 目录结构，保留共同 Git 祖先 |

私有备份是独立 GitHub 仓库，不属于原 fork 网络；保留共同 Git 历史使同步、比较和挑选提交仍然可行。最终 PR 的路径是：**整理后的代码 → 原公开 fork 的新分支 → 学长上游**。

0918 历史锚点：

- P1 seed2 基点：`077fbcab15b27efea779cbf0598f21f23987456b`。
- 框架及 selector 集成修正：`27d8ef8`。
- MID、factorial、A–F 和 owned runtime 代码备份：`238176dd44c3f192c697805e31224e2ab197cf71`。
- 原始阶段标签：`paper-checkpoint-20260918`，保留不移动。

这些是定位用的历史记录，不意味着以后只提交这两个新 commit。DiffusionDrive / V3 / P1 的改动可能更早就存在；必须审阅**相对于最新上游的完整差异**，不能漏掉前序依赖。

## 3. 每次开始先确认最新状态

执行者先检查用户当前代码目录、Git 状态、当前实验与最新私有备份。0918 分支只是旧检查点，不能假设它包含以后新增的成果。

1. 定位最新实现与相关测试、配置，区分已提交和未提交文件；先备份尚未保存的成果。
2. 明确本次是“仅私有备份”“同步上游”“准备可审阅 PR”还是“公开推送并提交 PR”。仅要求阅读本指南或备份不等于要求公开内部研究材料。
3. 核对仓库 URL、可见性、默认分支和 Git 共同祖先。若上游迁移或用户调整合作方式，以核实后的情况为准。
4. 不操作正在跑实验的工作树，不复制完整数据/模型/环境进 Git，不触碰学长服务器文件。

## 4. 建立独立的整理目录

以下命令是未来执行示例；目录已存在时先检查，不覆盖。新目录放在个人 `global_user` 下。

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry
git clone https://github.com/Ryannn-ry63/WorldEngine-paper-backup.git WorldEngine-pr-work
cd WorldEngine-pr-work
git remote add fork https://github.com/Ryannn-ry63/WorldEngine.git
git remote add upstream https://github.com/OpenDriveLab/WorldEngine.git
git fetch origin
git fetch fork
git fetch upstream
git remote set-head upstream -a
git remote -v
git symbolic-ref --short refs/remotes/upstream/HEAD
```

已有远端时核对 URL，不重复添加。下面的 `upstream/HEAD` 指向刚查到的上游默认分支，不硬编码其一定叫 main。

## 5. 同步学长的更新

先把最新研究实现整理为保留历史的代码分支，再建立独立同步分支。下面以 0918 代码为例；实际执行时换成确认过的最新分支。

```bash
git switch -c work/upstream-sync-0918 origin/research/paper-code-20260918
git merge upstream/HEAD
```

出现冲突时逐项核对上游行为与本地功能，保留必要的接口兼容性；不机械地整文件选择 ours/theirs。合并后检查变更范围，运行受影响功能的测试和小规模验证。历史实验仍对应原冻结代码，不把旧结果标成新代码所得。

同步分支通过验证后，可以保存到私有 `origin`；这一步本身不向公开 fork 或学长仓库发布任何内容。

## 6. 整理面向上游的最小完整贡献

从**最新上游**建立新的 PR 分支，不直接把私有快照 main 合并过去：

```bash
git switch -c pr/selector-paper-v1 upstream/HEAD
```

由执行者结合最新完整差异，把已审阅的代码按依赖顺序移入。可以挑选边界清晰的提交；混合了内部材料的提交应拆分整理，不能把整个研究历史批量 cherry-pick。必要时逐文件移入并重新形成清晰提交。

建议按可审阅的功能划分：DiffusionDrive 集成（如上游尚无）、V3 后训练、P1 方法及反馈支持、必要的评测入口、测试和公共运行文档。MID 是否进入正式方法，以最终论文结论为准。保留源码许可证及来源说明。

检查差异：

```bash
git diff --stat upstream/HEAD...HEAD
git diff --name-status upstream/HEAD...HEAD
git diff --check upstream/HEAD...HEAD
git log --oneline upstream/HEAD..HEAD
```

重点检查：

- 学长已有功能没有被无关修改、重命名或删除；必要的框架改动可以单独审阅。
- 没有密钥、个人绝对路径、内部方案/结果、迁移脚本、运行日志、无用 stage 和环境二进制。
- 配置和文档足够别人理解与运行；外部数据/权重通过明确配置提供。
- 新方法、原有入口兼容性与受影响的评测有适当验证；记录实际测试结果和未覆盖范围，不把已有 H100 检查当作新合并代码已验证。
- PR 描述围绕最终问题与行为变化，说明方法、必要框架修改及验证；文档、配置和实际实现一致。

这一步应先产出具体可审阅的代码与说明，再处理仍然需要用户决定的公开范围，不因泛泛的“以后要开源”而提前上传内部材料。用户已明确授权本次公开推送/提 PR 时，按其授权完成，不重复询问同一许可。

## 7. 推回原 fork，然后向学长提 PR

代码与公开范围确定、检查通过后，将**PR 分支**推到原 fork，不覆盖 fork 的主分支，也不 force push：

```bash
git push -u fork pr/selector-paper-v1
```

在 GitHub 选择：

- base repository：`OpenDriveLab/WorldEngine`；base branch：步骤 4 核实的上游默认分支。
- head repository：`Ryannn-ry63/WorldEngine`；compare branch：`pr/selector-paper-v1`。

若已安装并登录 GitHub CLI，可用：

```bash
UPSTREAM_BRANCH=$(git symbolic-ref --short refs/remotes/upstream/HEAD)
UPSTREAM_BRANCH=${UPSTREAM_BRANCH#upstream/}
gh pr create --repo OpenDriveLab/WorldEngine \
  --base "$UPSTREAM_BRANCH" \
  --head Ryannn-ry63:pr/selector-paper-v1 \
  --title '填写最终贡献标题' \
  --body-file /tmp/worldengine_pr_body.md \
  --draft
```

先把真实 PR 描述写到该 body 文件。没有 `gh` 时可用网页或 GitHub API，不因此重建仓库。默认以 draft 方便学长审阅；用户另有要求时按要求执行。最终核对远端提交、base/head、实际 diff，并回报 PR 链接与验证结果。

## 8. 完成后的记录

保存私有备份提交、公开 fork 分支/提交、上游基点、PR 链接、测试结果及剩余限制。后续修订在对应 PR 分支进行，依照授权推送到 fork 即可更新同一个 PR。

不要移动 0918 原始标签。后续阶段新建有日期的标签；更新快照文件时同步刷新 `registry/backup_inventory.json`，核对文件 SHA256。GitHub 代码备份不包含数据、权重、完整缓存和 Conda 环境，不能替代这些文件的独立保存。
