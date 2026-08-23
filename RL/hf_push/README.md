---
license: mit
tags:
- speculative-decoding
- eagle3
- energy
- gpu
- sglang
pretty_name: EAGLE3 Speculative Decoding Energy Sweep
---

# EAGLE3 Speculative Decoding Energy Sweep

Per-config energy/throughput/latency measurements for EAGLE3 speculative decoding
(`speculative_num_steps`, `speculative_eagle_topk`, `speculative_num_draft_tokens`)
served with sglang, across batch sizes. Collected for an RL project that learns to
pick speculative-decoding parameters to hold GPU energy utilization in a target band.

Model: `unsloth/Llama-3.2-1B-Instruct` + `rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct` draft head.
Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8GB).

## Files

- `eagle3_energy_sweep.jsonl` — one row per `(batch_size, steps, topk, num_draft_tokens, repeat)` config, 3 repeats of the full grid (batch sizes 1, 4, 8, 16), plus 12 agentic pilot-trial rows appended at the end (see below).

## Agentic pilot traces (rows 228-239)

228-239 are not part of the synthetic sweep grid. They're individual trials from two
agentic pilots that checked whether the sweep's speculative-decoding picks still help
on real tool-use tasks: a 2-instance SWE-bench Lite patch-generation pilot and a
2-task Terminal-Bench (Harbor, Terminus 2 agent) pilot, each run at 3 configs
(`no_spec`, `chosen`=(3,4,8), `chosen_bs16`=(3,2,4)). `speed_source` is
`"swebench_pilot"` or `"terminalbench_pilot"` for these rows (vs. `"step_time"` for
the sweep grid).

These measure a fundamentally different thing than the sweep grid: end-to-end
wall-clock of a multi-turn tool-use agent session (or, for SWE-bench, a single
long-context patch-generation request), dominated by non-LLM overhead — tool
execution, JSON-parse retries, and in one case (`chosen_bs16` on Terminal-Bench's
`regex-log` task) a runaway generation loop that burned ~895s of a 900s timeout on a
single stuck call. That is not comparable to the sweep's controlled, fixed-length
synthetic decode throughput.

To keep them from corrupting anything downstream: `is_baseline` is always `false`
for these rows (even the `no_spec` ones), and `error` is always a non-null string
(`"agentic_pilot_trace: ..."`) rather than a real failure. Both fields are set this
way *by design*, not because the underlying trial errored — this repo's training
code (`RL/train_offline.py`, `RL/algos/common.py`) already skips any row with a
truthy `error` and uses `is_baseline` rows to compute the per-`batch_size` reference
throughput used to normalize every reward, so these two fields are the mechanism
that excludes the 12 rows from bandit/offline-RL training while still keeping the
real measured numbers (tok/s, energy, GPU stats, `avg_spec_accept_length`) in the
file for anyone doing manual analysis. `batch_size` is `1` for all 12 rows — that's
the actual concurrency both pilots ran at (one trial at a time), even for the
`chosen_bs16` config whose name refers to the sweep-grid batch size it was chosen
at, not the concurrency it was tested at here. `gpu_temp_c_before`/`gpu_mem_used_mb`/
`gpu_throttling` are `null` for the Terminal-Bench rows (that pilot's telemetry is a
continuous sampler segmented by trial window, which doesn't produce a distinct
"before" snapshot or capture memory/throttling in its current report format); the
SWE-bench rows have all of these directly, from before/after NVML snapshots
bracketing each request.

## Retries

Some configs failed to launch under sglang's initial `--mem-fraction-static 0.65`
(the speculative-tree buffers didn't fit at higher `steps`/batch size). Those were
rerun at `--mem-fraction-static 0.55`; where the retry succeeded, its row replaces
the original error row (same `index`, `retry_mem_fraction` set). The single hardest
config in the grid — `batch_size=16, steps=5, topk=4, num_draft_tokens=16` (the
largest possible speculative tree) — still fails to launch even at 0.55 and is left
as an error row; its VRAM footprint just doesn't fit in 8GB at that batch size.

## Columns

| column | meaning |
|---|---|
| `index`, `repeat` | position in the sweep grid / which repeat pass |
| `batch_size`, `steps`, `topk`, `num_draft_tokens` | the speculative-decoding config under test |
| `is_baseline` | `true` for the non-speculative `(0,0,0)` config |
| `error` | failure message if the config didn't run; `null` on success |
| `retry_mem_fraction` | set if this row came from a retry at a lower mem-fraction |
| `accept_length` | mean accepted draft tokens per step |
| `speed_tok_s`, `speed_source`, `step_time_s` | decode throughput |
| `wall_clock_s`, `wall_clock_tok_s`, `total_output_tokens` | end-to-end request timing |
| `energy_joules`, `avg_power_watts`, `joules_per_token` | energy accounting |
| `gpu_temp_c_before`, `gpu_temp_c`, `gpu_power_w`, `gpu_util_pct`, `gpu_mem_used_mb`, `gpu_throttling` | GPU telemetry during the run |
