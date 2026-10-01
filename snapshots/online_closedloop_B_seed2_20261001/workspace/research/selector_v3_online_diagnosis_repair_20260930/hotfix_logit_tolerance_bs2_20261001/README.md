# B_s2 同版数值检查修复

B_s2 在 round101 rank4 因 `Training/inference selector logits diverged: 1.3828277587890625e-05` 中止。已提交 808/904 episodes；最后完整轮的 8 个 checkpoint 和 8 个 report 哈希均核验通过。

复用 A_s2/B_s1 已在 H100 跑完 904 episodes 的数值补丁，全部 overlay 文件逐字节一致，仅独立入口的任务白名单改成 B_s2。原热修复目录保持不变，原注册、checkpoint、设置、奖励和优化公式不改。失败记录与续跑前进度已保存在 evidence，正式恢复继续使用原锁与 campaign deadline。

在一台 8×H100 上运行（前台输出并保存日志；自动启动预检）：

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930/hotfix_logit_tolerance_bs2_20261001/start_hotfix_job_h100.sh train_B_s2
```

恢复 808 后继续剩余 96 episodes，重做未提交轮。无需重训另外五个已完成任务。

完整 CHECK_ONLY 预检退出码 0；数据校验、AlgEngine/SimEngine 导入及 PASS_NUMERIC_HOTFIX 实际补丁路径检查均通过。尚未在本地启动训练。详见 evidence/validation.json 与 logs。
