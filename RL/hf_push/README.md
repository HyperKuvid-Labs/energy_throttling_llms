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

- `eagle3_energy_sweep.jsonl` — one row per `(batch_size, steps, topk, num_draft_tokens, repeat)` config, 3 repeats of the full grid (batch sizes 1, 4, 8, 16).

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
