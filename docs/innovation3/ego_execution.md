# Ego execution aligned with upstream closed-loop evaluation

Updated 2026-09-26. The online default is now `log_play_controller`, matching
`projects/AlgEngine/scripts/run_ray_distributed_testing.sh`. It directly applies
the next trajectory waypoint, velocity, heading and angular velocity every 0.5 s.
The main branch and all counterfactual branches share this controller.

- Runtime setting: `online_ego_controller: log_play_controller` (also the default).
- Execution identifier: `upstream_log_play_h1_v1`.
- Historical optional controller: `two_stage_controller`, identifier
  `online_signed_accel_steering_rate_substeps_v3`.
- Generated actions keep the upstream rear-axle-to-center conversion and 0.1 s
  velocity calculation before 0.5 s downsampling. Map warmup and diagnostic
  requests now supply complete velocity fields on their own 0.5 s grid.
- Reports, worker receipts and new checkpoints identify the execution protocol.
  The parent verifies the worker protocol. Resume rejects missing/different
  execution identifiers and changed source hashes. Start a new run from offline
  V3 when changing controllers; previous experiment records remain historical.

The H1 reward formula, full20 branching, strict snapshot/reward/history parity,
selector-only updates and 13-step map warmup remain in place. This aligns ego
execution, not the entire training reward or initialization with evaluation.
Ideal tracking remains closed-loop: the executed pose changes the next rendered
observation. NR/R describes surrounding traffic, not a different ego controller.

Do not infer runtime speedup from removing LQR/bicycle integration. Rendering,
model inference, snapshots, reactive traffic and reward scoring remain. A new
visual/GPU acceptance run and controlled throughput comparison are still needed.

Behavior tests: `tests/innovation3/test_log_play_execution.py`,
`test_candidate_bridge.py`, `test_headless_snapshot.py`, `test_parallel_branches.py`,
`test_causal_reward.py`, `test_warmup_runtime.py`, `test_online_probe_driver.py`.
