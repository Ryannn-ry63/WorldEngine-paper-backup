# 数据和结果迁移后的运行命令（2026-09-18）

范围：A–F六个队列 + prior旧完整闭环续跑，没有G。现在仍使用原data；本次没有删除、改写或重建data软链接，也没有启动正式实验。

## 一次性确认

学长迁好公共data后，仍使用原逻辑入口：
`/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/data`

`data`里面的二级软链接也必须可访问，特别是raw图像、nuplan maps、仿真assets和 `sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl`。仅顶层链接有效不代表这些目标已迁好；检查器会逐项检测，不自行复制/修改公共data。

在目标实例执行：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-paper/code
bash run_owned_tasks.sh audit all
```

此命令不使用GPU、不启动训练/评测，只核对七项任务的有限输入元数据、依赖位置和数据端点。迁移未结束时E等任务可能显示缺少collection清单，待迁完重试。通过为各项 `PASS_SCOPED_PATH_AUDIT`；它不是全量文件哈希或正式实验通过标志。

需要在目标实例预检环境/GPU时：

```bash
bash run_owned_tasks.sh check A --device 0
```

该检查仍可在原data上做，不代表新data已迁好；正式start默认拒绝任何data端点仍指向原roboticsystem2项目。用户确认继续使用原项目data时可显式加 `--allow-original-data`，但不能绕过结果迁移锁和完整校验门槛。

## 七个独立四卡任务

每条命令使用独立的四卡实例，至少36 CPU核。A已完成的结果会被复核复用。B/C/F先核验所有训练、优化器端点与native模型，直接续评测；D继续其未完成训练，E继续分支诊断。prior是原先 `run_h100_full_eval.sh` 对应任务的恢复入口。

```bash
# 原完整闭环：固定V3与D1–D4，补齐full/common、NR/R
bash run_owned_tasks.sh start prior

# A：旧模型开环表格、oracle和来源审计
bash run_owned_tasks.sh start A

# B：反馈目标消融，39个训练端点已完成，续评测
bash run_owned_tasks.sh start B

# C：结构消融，24个训练端点已完成，续评测
bash run_owned_tasks.sh start C

# D：监督对照，继续未完成训练并评测（沿用四卡H100）
bash run_owned_tasks.sh start D

# E：full20物理分支诊断
bash run_owned_tasks.sh start E

# F：三种数据来源基线，9个训练端点已完成，续评测
bash run_owned_tasks.sh start F
```

不要将整个代码块一次性提交到一个实例；每个实例选一条。四卡默认0,1,2,3，可显式 `--devices 0,1,2,3`。不再直接使用裸 `run_paper_study.sh` 或旧full_eval入口，避免漏掉自有路径映射。

D原训练记录绑定 `NVIDIA H100 80GB HBM3` / torch 2.0.1+cu118。保留其恢复身份检查，不能把H200训练伪装成H100。D planner没有已保存checkpoint的那段训练会按原驱动从头开始；有latest.pth才恢复优化器。评测入口支持H100/H200，原四卡CUDA预检仍执行。单卡通过并不代表四卡DDP已经验收。

## 启动时具体做什么

- 环境来自已有个人algengine、已恢复的个人simengine，以及WorldEngine-owned中的依赖备份、基础模型；旧前缀在进程内部映射到这些个人文件，维持冻结记录的路径身份。
- 在进程视图隐藏旧project目录，只重新开放明确的个人依赖和data端点。非data实际存储必须在 `/inspire/hdd/global_user/wangcaojun-240208020180/`；data是唯一实验文件例外。操作系统、NVIDIA驱动、系统动态库等仍由实例镜像提供。
- 每次根据当前data链接解析真实位置，再兼容历史绝对路径，包括旧full_scenario路径；不把今天的旧data位置写死，不改已有契约/指标/模型哈希。
- 检查后台迁移锁及 `PASS_BYTE_COPY_ONLY`；正式任务整个运行期持有共享锁，阻止复制器重新写入同一批结果。
- 重做本实例自有GPU依赖检查，再调用原生产驱动。保留冻结输入、模型哈希、runtime、GPU类型、资源契约和评测预检。
- B/C/D/F中没有audit/accepted的中断闭环条件，先检查身份、获取锁并完整归档现场，之后从该条件开头重跑；已审计内容不覆盖。E和prior沿用原来的现场保留与复用逻辑。不会承诺所有中断条件都能逐帧继续。
- 任务输出和reports实际写入 `WorldEngine-owned/project_outputs/WorldEngine-paper`（或原本已在个人global_user的历史目录）。旧逻辑路径可能仍出现在日志中，不表示写回旧项目。

## 已验证与待验证

已完成：原data+个人依赖的单卡H100真实NR/R历史一致性与结果复用（33分15秒）；新入口在隐藏project目录的视图下CPU导入/实际来源检查；data原地换链解析测试；嵌套断链/非data越界拒绝；失败现场保留/已审计结果保护；迁移进行中拒绝正式启动；冻结paper代码、原评测驱动和旧full_resume驱动哈希未变。

尚待：全量结果复制/校验结束、E尚未迁来的清单和产物、新项目data路径和数据内容验收、目标实例四卡/训练恢复。不能提前标记这些已通过。每次运行打印 `TASK_CHECK_DIRECTORY`，详细检查结果在 `WorldEngine-owned/runtime/task_checks/`。
