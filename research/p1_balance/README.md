# P1 balance：双域小对照与三 seed 原生评测

这是已批准的下一轮实现，不是新颖性结论。历史 P1、CL_NLL、V3、分析 run 保持不变。
新工作树基于 attribution 的 077fbca，并保留该工作树当时未提交的依赖源码；未自动提交或推送。

## 固定实验

- 小对照：CL_NLL、MID、P1、MID_SCALE_MATCH；新增 MID 两组各 seed0/1/2，共6模型。
- 从对应 shared 父节点恢复 optimizer/RNG，继续500步，原256状态/20候选/两个已查询候选。
- 全20 winner NLL + λPair，λ=0/.5/1；GRPO、teacher KL、学习率、采样不变。
- MID_SCALE_MATCH 的系数从初始父节点、完整固定训练批次计算并冻结；只匹配初始反馈梯度RMS。
- 开发评测48条件：24复用+24新增，rare58/common64、NR/R、旧开发噪声。
- 正式评测42条件：12复用+30新增，P1/CL_NLL各三 seed 加固定V3。
- 每模型覆盖 open-navtest12146、open-rare288、full-NR288、full-R288、common-NR64、common-R64。
- 五个新增原生模型先做bridge8的NR/R，共10条件。仅替换54个selector张量，其余963个张量不变。
- 正式噪声 formal_navtest_seed0。common64不是全量common；历史暴露不因此消失。

参数固定，不挑中途checkpoint或最好seed，不采集新标签，不重启P3，不自动评估MID全量。
工程PASS不代表性能通过；全部结果保留，终点始终REVIEW_REQUIRED。

## 两个独立任务（2026-09-14 拆分）

用户决定将三 seed 评测提交平台独立运行。方法、种子、数据、噪声与复用规则不变；
只拆执行范围、预算和完成条件。下列新入口替代本轮合并 `all` 命令。

| task | 工作 | 新增工作 | 默认 run ID | GPUh 上限 |
|---|---|---|---|---:|
| development | MID 小对照 | 6 模型训练 + 24 开发评测 | p1_mid_control_20260914_split_v1 | 64 |
| formal | P1/CL_NLL 三 seed 原生评测 | 10 bridge + 30 正式评测 | p1_3seed_formal_20260914_split_v1 | 128 |

两任务各申请单节点 4 张 H100，分别启动一个脚本进程；内部调度四卡。
可以放在不同实例同时运行。不使用 torchrun、不设置每 GPU 一个脚本副本；
当前入口不支持一个实验任务跨多个节点。相同主机上的 balance GPU runner 仍互斥。

平台启动命令（三 seed 任务）：

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-p1-balance-v1/run_diffusiondrive_selector_p1_task_4h100.sh formal
```

小对照在另一四卡实例运行：

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-p1-balance-v1/run_diffusiondrive_selector_p1_task_4h100.sh development
```

平台任务直接前台执行上述命令，不加 nohup 或 `&`。启动脚本先 audit，再执行范围内的
preflight、任务队列和报告；任一阶段失败就退出。重复同一条命令可恢复，同 ID 的 task、
代码和预算均不可修改。可选位置参数是 `[run_id] [gpu_hours]`。
若实际 SimEngine 环境与默认路径不同，提交前配置 `DIFFUSIONDRIVE_SIMENGINE_PYTHON`。
实例需要能访问原工作区、数据和两个历史 source run。

历史预测拆分后：development 约 5.44 小时实测组件量，加余量约 8.34 小时；
formal 约 15.67 小时实测组件量，加余量约 22.15 小时。
每任务各加 35% 和 4 GPUh 辅助余量，所以两者规划时间之和略大于原合并预测。
这些是历史速度外推，CPU audit/导出也有启动成本；不是保证时长。
64/128 GPUh 分别对应四卡 16/32 小时累计记账上限，总上限仍为 192 GPUh。

输出均位于 `experiments/diffusiondrive/selector_p1_balance_v1/runs/<run_id>/`，
各有独立 contract、ledger、forecast、summary.md/json、paired_scenes.csv 和 release_manifest。
formal 另有 29 列 summary.tsv。每条任务按自身范围判定工程 PASS，不等待另一条任务；
最终科学结论仍为 REVIEW_REQUIRED。formal 不启动 MID 训练或开发 rollout，也不计算 MID
强度系数；development 不启动原生 bridge 或正式 rollout。
两边仍共用原有历史输入审计与 CPU 导出/一致性检查，以保持来源核验一致。

源码加入 task 范围后，必须使用上述新 run ID。旧软件审计与旧 run 的冻结归档保留，
不要拿旧 `software_audit_balance_20260914_v1` 的审计直接运行新代码。
正式任务启动后保持本工作树源码不变；需要开发后续方法时使用另一工作树。

只做 CPU audit 或离线汇总时可使用底层入口，例如：

```bash
bash run_diffusiondrive_selector_p1_balance_4h100.sh audit p1_3seed_formal_20260914_split_v1 --task formal --gpus 4 --gpu-hours 128
bash run_diffusiondrive_selector_p1_balance_4h100.sh report p1_3seed_formal_20260914_split_v1 --task formal --gpus 4 --gpu-hours 128
```

## 原合并运行方式（保留参考）

所有命令在新工作树执行，不要运行旧 attribution 或 seed2-only snapshot 入口。

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-p1-balance-v1
bash run_diffusiondrive_selector_p1_balance_4h100.sh audit p1_balance_20260914_v1 --gpus 4 --gpu-hours 192
```

audit是CPU操作：冻结输入与代码、导出五个端点、检查复用和原生输出、计算强度系数、估算预算/存储。
不训练、不运行闭环。不要在源代码仍在修改时使用正式run ID。

四卡实例上先预检，再后台自动完成两条线：

```bash
bash run_diffusiondrive_selector_p1_balance_4h100.sh preflight p1_balance_20260914_v1 --gpus 4 --gpu-hours 192
nohup bash run_diffusiondrive_selector_p1_balance_4h100.sh all p1_balance_20260914_v1 --gpus 4 --gpu-hours 192 > p1_balance_20260914_v1.log 2>&1 &
tail -n 40 p1_balance_20260914_v1.log
```

all：预检→6模型训练→10项bridge→按seed穿插开发与正式闭环→正式开环→报告。
每个任务同一时刻只占用四卡；同一主机不同balance run也互斥，不能开两个四卡runner。
不因效果下降跳过后续条件，不自动追加实验。

192 GPUh是累计上限，最多相当于四卡48小时，不是预计时长。具体估时见audit的budget_forecast.json。
报告保留历史实测依据与余量，失败重试另耗预算。

中断后确认旧进程退出，再用相同代码、ID和参数恢复：

```bash
nohup bash run_diffusiondrive_selector_p1_balance_4h100.sh all p1_balance_20260914_v1 --gpus 4 --gpu-hours 192 >> p1_balance_20260914_v1.log 2>&1 &
```

训练每25步保存模型/optimizer/NumPy/Torch/CUDA RNG；不完整评测按条件归档后重跑，完整条件核验后复用。
不要删除ledger重置预算，不要覆盖历史结果。代码/输入/预算改变使用新run ID。

单独阶段：train、dev-eval、native-bridge、formal-eval；formal-eval要求全部bridge通过。
离线汇总（不启动GPU，须无同run活动runner）：

```bash
bash run_diffusiondrive_selector_p1_balance_4h100.sh report p1_balance_20260914_v1 --gpus 4 --gpu-hours 192
```

如实例 SimEngine Python 不在默认 /root/miniconda3/envs/simengine/bin/python，
先设置 DIFFUSIONDRIVE_SIMENGINE_PYTHON 为实例上的真实路径。

## 输出与判断

产物在 experiments/diffusiondrive/selector_p1_balance_v1/runs/对应runID/：
audit_report.json、scale_audit.json、cached_parity_cpu.json、budget_forecast.json、bridge_report.json、
train/、collections/、openloop/、progress.json、summary.md/json、paired_scenes.csv、archive/和ledger.json。

开发/正式结果分开；报告全部seed、均值/标准差、安全分量、进度、rescued/broken、V3定义的难例分层。
rare288单列development58/remainder230；不是盲测。先平均seed，再按log做10000次bootstrap，
附equal-log与leave-one-log-out；区间条件于固定采集/噪声，未做多重比较校正。

MID四项PDM均值不低于P1且至少一项提高，仅标为均值层面候选；不是显著性或非劣性证明。
简单调权有效时作为强化基线；仍有取舍也不能宣称已证明因果根因。
后续方法需要独立训练侧证据，不能把本轮评测失败案例直接用作训练标签。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 /inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python -m unittest discover -s research/p1_balance/tests -v
```

CPU和合成测试不能替代四卡CUDA与真实bridge；实施阶段不自行启动正式GPU实验。
