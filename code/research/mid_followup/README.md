# MID follow-up v1

Primary model: existing MID lambda=0.5, seeds 0/1/2. This worktree is isolated from the original running P1/CL_NLL task. `source_snapshot.json` records the 53 inherited uncommitted source files.

| Task | New work | Reused | Four-H100 budget |
|---|---|---|---|
| analysis | CPU diagnostics and executed-case analysis | Completed MID/P1/CL_NLL/MID_SCALE_MATCH | No GPU reservation |
| sensitivity | lambda=0.4/0.6 × 3 seeds: 6 training jobs, 24 development evaluations | 48 development evaluations | 48 GPUh; roughly 8–9 hours |
| mid-formal | 6 bridges, then 18 native evaluations for existing MID seeds 0/1/2 | 6 scalar V3 evaluations | 72 GPUh; roughly 14 hours |

Each task owns its contract, cumulative budget ledger and output directory. Forecasts use completed source timings plus 35% and four GPUh overhead. No automatic coefficient selection, acquisition, expansion or promotion; complete reports retain `REVIEW_REQUIRED`.

Sensitivity changes only lambda in `dense_GRPO + winner_NLL + lambda*pair + 0.01*teacher_KL`. All jobs restore the original shared step-500 parent, optimizer and NumPy/Torch/CUDA RNG, and take 500 further steps. Original 20 candidates, 256 feedback states, sampling, clipping and optimizer are fixed. MID_SCALE_MATCH remains a diagnostic; it is excluded from the lambda curve because it changes global feedback scale.

Formal evaluation replaces only 54 selector tensors and retains 963 nonselector tensors. It requires CPU parity, CPU/CUDA parity and each model's NR/R bridge against its original development rollout. Six blocks: open-navtest, open-rare, full288 NR/R and common64 NR/R. Common64 is not full-common. Formal noise is `formal_navtest_seed0`; bridge/development use the original namespace.

## Run commands

Working directory for every command:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-mid-followup-v1
```

Submit each command as **one process on one node with four H100 GPUs**, one distributed task per command. The launcher owns internal workers; do not wrap in torchrun/mpirun or start four copies. Different nodes can run both tasks concurrently. Scheduler CUDA_VISIBLE_DEVICES is preserved; the shared host lock prevents two queues sharing an allocation.

```bash
bash run_diffusiondrive_selector_mid_followup_4h100.sh mid-formal mid_formal_20260915_v1 72
```

```bash
bash run_diffusiondrive_selector_mid_followup_4h100.sh sensitivity mid_sensitivity_20260915_v1 48
```

Default stage `all` audits first, checks four-H100 hardware and then executes. On interruption rerun the **same command/run ID**: audited results are reused and training resumes from its last saved 25-step checkpoint. GPU time remains cumulative, including retries and helpers. Never delete/reset ledgers or edit frozen code/contracts. Changed code/budget needs a new run ID.

CPU audit and report-only stages:

```bash
bash run_diffusiondrive_selector_mid_followup_4h100.sh mid-formal mid_formal_20260915_v1 72 audit
bash run_diffusiondrive_selector_mid_followup_4h100.sh sensitivity mid_sensitivity_20260915_v1 48 audit
bash run_diffusiondrive_selector_mid_followup_4h100.sh mid-formal mid_formal_20260915_v1 72 report
```

AlgEngine environment root defaults to `/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine`; override via DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE. SimEngine Python defaults to `/root/miniconda3/envs/simengine/bin/python`; override via DIFFUSIONDRIVE_SIMENGINE_PYTHON if needed. Production preflight checks CUDA extensions and both environments. CPU audit is not production GPU preflight.

Run analysis on the instance, using a new empty output directory:

```bash
bash run_diffusiondrive_selector_mid_followup_4h100.sh analysis \
  --output experiments/diffusiondrive/selector_mid_followup_v1/analyses/mid_analysis_20260915_v1
```

After sensitivity completes, use a new output directory and add `--sensitivity-run <absolute run path>`. To join the completed full results, also pass both `--mid-formal-run <absolute MID run path>` and `--reference-formal-run <absolute P1/CL_NLL full run path>`. Both full runs must have complete released reports and stopped ledgers. MID never waits for partial reference results or changes its settings based on them.

## Evidence and outputs

GPU outputs: `experiments/diffusiondrive/selector_mid_followup_v1/runs/<run_id>/`. Key files: audit_report.json, budget_forecast.json, ledger.json, progress.json, completed summary.json/summary.md, archive/release_manifest.json. Sensitivity adds sensitivity.png (individual seeds and mean ± seed std), curve CSV and paired scene CSV. Formal adds a 29-column TSV and paired scene CSV. Negative results and all seeds remain included.

Analysis outputs: statistics.json, training_probes.json, decision_cases.json, formal_comparison.json, input_manifest.json and review.md. All 256 training states are scored; gradients cover all 16 feedback batches with one fixed dense batch per seed. Pair ranking, pair mass, winner selection, outside selection, safety strata and weighted feedback/total gradients are separated. No optimizer is constructed and parameters are checked unchanged. Shared-parent combined gradients are explicitly labeled as the P1 reference objective, not the parent training objective.

Cases include all per-seed safety disagreements plus top three positive/negative independent logs per domain/mode/MID comparison. All models are cross-scored on both policies' recorded contexts. Comparable first action divergence requires physical pre-action state, visible context and a shared executed prefix. A broken prefix cannot rejoin. Later states are not counterfactual outcomes; missing violation times remain NOT_RECORDED. Evaluation cases never become new training labels.

Statistics average seeds before 10,000 log bootstrap samples. Intervals are exploratory and unadjusted, conditional on fixed acquisition/noise; they are not noninferiority proof. Full288 results separately report development58 and remainder230. All sets retain historical-exposure qualifications; no independent unseen-test claim is made. Open-loop seed summaries do not invent log mappings.

A full native table is part of a paper suite. Mechanism, stronger baselines and independent settings still require research; a loss-weight change alone is not a novelty claim.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 /inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python -m unittest discover -s research/mid_followup/tests -v
PYTHONDONTWRITEBYTECODE=1 /inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python -m unittest discover -s research/p1_balance/tests -v
```

Hashes are reused only within one process while device, inode, size, mtime and ctime match. Files modified in the last two seconds are never cached, to cover coarse filesystem timestamps. Changed files are re-read and checked against the expected digest. Final analysis model hashes bypass this cache.

Tests cover fixed losses/gradients, task scope, exports, physical prefix rules, immutable evidence, interruption/resume equivalence, reports, budget forecast and unattended launcher errors. Real CPU audit and GPU bridges remain necessary.
