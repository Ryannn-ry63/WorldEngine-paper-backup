# 保留上游同步与未来 Pull Request

## 关系不变

现有公开仓库 `https://github.com/Ryannn-ry63/WorldEngine` 是 `OpenDriveLab/WorldEngine` 的fork。这个fork网络保持不变；私有备份是独立仓库，不会出现在同一个GitHub fork网络里，但代码历史分支保留共同祖先，可正常fetch/merge/cherry-pick。

`research/paper-code-20260918` 基于已发布的P1 seed2提交 `077fbcab15b27efea779cbf0598f21f23987456b`；不是无祖先的重建代码。`main`是独立工作区快照，不作为向学长提交PR的分支。

## 添加远端（在私有备份clone中）

```bash
git remote add fork https://github.com/Ryannn-ry63/WorldEngine.git
git remote add upstream https://github.com/OpenDriveLab/WorldEngine.git
git fetch upstream
```

若远端已存在，先 `git remote -v` 检查，无需重复添加。不要更改原工作区正在运行实验的分支或冻结源码。

## 后续同步上游

在独立checkout中，从代码历史分支新建工作分支，再合并需要的upstream更新，检查冲突并做行为测试。不要在冻结实验运行期间直接更新其runtime，也不要把原协议的结果重新标为更新后代码所得。

```bash
git switch research/paper-code-20260918
git switch -c work/upstream-sync
git merge upstream/main
```

以上为后续工作示例，本次没有替用户执行合并或改写公开分支。

## 未来提交给学长

另从当时的 `upstream/main` 新建PR分支，按功能挑选、整理并测试代码改动，然后推到现有公开fork，由fork向上游提PR。私有备份分支不能直接跨GitHub仓库网络提交PR；代码经审阅后推至原fork即可恢复原有PR路径。

不要将 `main`中的内部报告、个人路径注册表、迁移诊断、全部历史探索直接合并进上游。框架修改与方法新增分开审阅。当前代码分支保留原README和原文件；相对于P1 seed2，本次没有删除原文件，已有文件的修改单列在 `registry/code_branch_changes.json`，包括仿真入口的必要适配。相对于学长的纯上游，此前已存在的DiffusionDrive集成仍需纳入最终PR审阅，不能误称“完全没改上游”。

## 0918 完整操作指南

后续执行以 [20260918_GITHUB_UPSTREAM_PR_GUIDE.md](20260918_GITHUB_UPSTREAM_PR_GUIDE.md) 为详细指南，涵盖最新代码核对、独立整理目录、同步上游、贡献拆分、推回原 fork、创建 PR 及验收记录。
