# Energy-Aware Speculative Decoding Control

This is meant to work as a cookbook, not a paper. The actual deliverable is a
small table of `(speculative_num_steps, speculative_eagle_topk,
speculative_num_draft_tokens)` values per batch size that are known-good on
this GPU, so speculative-decoding params for EAGLE3 + sglang don't have to be
guessed by hand every time a server gets spun up on the laptop. Everything
else here -- the sweep, the reward, the eight algorithms -- exists to earn the
numbers in the table below.

The target: hold GPU energy utilization (average power draw / power limit)
inside a tight 95-98% band. Below the band the GPU has idle capacity worth
spending on deeper speculation; above it the card is approaching its power
cap, clocks down, and more speculation buys nothing but costs energy.

## TL;DR -- the recipe

Live-measured on an RTX 4060 Laptop GPU (8GB), `unsloth/Llama-3.2-1B-Instruct`
+ `rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct` draft head, 80W power cap:

| batch_size | steps | topk | draft | live reward | vs fixed `(3,4,8)` reference |
|---|---|---|---|---|---|
| 1  | 3 | 4 | 8 | 0.5323 | same config, no change needed |
| 4  | 3 | 4 | 8 | 0.4975 | same config, no change needed |
| 8  | 3 | 4 | 8 | 0.4653 | same config, no change needed |
| 16 | 3 | 2 | 4 | 0.3855 | **+0.091**, avoids overshooting the energy band |

At `bs=1,4,8` there's nothing to gain over the reference config -- every
algorithm tested either reproduces it exactly or does worse. `bs=16` is the
one case worth actually changing the launch flags for, and four independent
methods (the MLP bandit, a plain per-bs lookup table, LinUCB, and doubly
robust) all converge on the same answer.

## What the policy takes in, what it puts out

**Input** -- a 4-dim state, read from GPU telemetry after a config has run
(not a live pre-decision sensor read, see caveat below):

```
[ batch_size / 8.0,
  gpu_temp_c_before / 100.0,
  gpu_mem_used_mb / 8192.0,
  gpu_util_pct / 100.0 ]
```

**Output** -- an index into a fixed, enumerated `ActionSpace` (`RL/policy.py`),
decoded back into the three sglang launch flags:
`--speculative-num-steps`, `--speculative-eagle-topk`,
`--speculative-num-draft-tokens`.

The full grid before filtering is `steps ∈ {0,1,3,5} × topk ∈ {0,1,2,4} ×
draft ∈ {0,2,4,8,16}` across `batch_size ∈ {1,4,8,16}`. Not every combination
is legal -- constraints pulled from sglang 0.5.2 source, not the docs
(`RL/sweep_config.py`):

1. `steps * topk + 1 >= num_draft_tokens` -- surplus draft tokens can never be
   filled, sglang's own bench script skips these.
2. `topk == 1` forces `num_draft_tokens = steps + 1` server-side (a logged
   warning, not an error), so two nominally different configs collapse into
   one and have to be deduplicated.
3. `(0, 0, 0)` is the special-cased non-speculative baseline; any other
   zero-containing combination is invalid.

After validity + dedup that's **19 distinct actions**, shared across all four
batch sizes -- batch size is a state feature, not part of the action.

Caveat worth keeping in mind: `gpu_mem_used_mb` and `gpu_util_pct` are read
from `metrics_after` in the sweep row, i.e. *after* that config already ran,
not a genuine pre-decision read. Every pick from every algorithm below
inherits this. Fine for choosing a static per-batch-size config (everything
tested here); would need fixing before trusting any of these to react to
live GPU state mid-session.

## The sweep -- how the data was collected

`RL/run_sweep.py` launches a real sglang server subprocess per
`(batch_size, config)` pair directly on the physical GPU. Nothing is faked
and no server is ever launched from inside a training loop -- the sweep runs
once, gets cached to JSONL, and every algorithm below trains against that
cache, free, on CPU.

Per config:

1. Launch sglang with the given speculative flags. Target and draft are
   forced to the same `dtype=float16` -- the EAGLE3 draft head ships fp16 but
   the target ships bf16, and sglang doesn't reconcile them, so CUDA graph
   capture dies with a dtype mismatch otherwise. Applied to the baseline too,
   so the comparison stays dtype-matched.
2. Poll `/health` until the server's up.
3. Fire one warmup batch.
4. Snapshot GPU energy (`nvmlDeviceGetTotalEnergyConsumption`) and telemetry
   before the real batch.
5. Fire `max(num_prompts, batch_size)` requests at concurrency = batch_size,
   against 4 fixed prompts, 512 output tokens each -- same methodology as
   sglang's own `scripts/playground/bench_speculative.py`, so the numbers are
   directly comparable to sglang's published ones.
6. Snapshot energy/telemetry again, read `/get_server_info` for
   `avg_spec_accept_length` and `step_time_dict`.
7. `speed_tok_s = accept_length / p20(step_time)` -- 20th percentile, same as
   sglang, less sensitive to warmup outliers. Falls back to wall-clock/bs if
   `step_time_dict` didn't populate.
8. Tear the server down, sleep 5s so the GPU settles before the next launch.

Reward is computed offline from the recorded row (`RL/reward.py`):

- `energy_utilization = avg_power_watts / power_limit_watts` (80W cap on this
  laptop).
- `band_score`: 1.0 inside `[0.95, 0.98]`, ramps linearly in from 0 below the
  band, falls off at a 10x slope above it.
- `throughput_score = clip((speed / baseline_speed - 1) / 2, 0, 1)` -- speedup
  over the non-speculative baseline, saturates at 3x, floors at 0. A direct
  consequence: any config that IS the baseline scores `throughput_score = 0`
  against itself, exactly, every time -- relevant below.
- `reward = (band_score^w * throughput_score^(1-w)) * thermal_multiplier`,
  geometric mean (`w=0.5` default) so neither term can be ignored, then a
  multiplicative penalty for `>75°C` (0.8x), `>80°C` (0.5x), or active
  throttling (0.3x). Clipped to `[0, 1]`.

Dataset: 228 rows, 3 full repeats of the 76-cell grid (4 batch sizes x 19
actions), landing at different thermal states per repeat since repeats aren't
run back-to-back per config. A handful of configs that failed to launch at
`--mem-fraction-static 0.65` were retried at 0.55; one config
(`bs=16, steps=5, topk=4, draft=16`) never fits in 8GB regardless and is left
as a permanent error row. Published at
[`Pradheep1647/eagle3-speculative-decoding-energy-sweep`](https://huggingface.co/datasets/Pradheep1647/eagle3-speculative-decoding-energy-sweep).

## Results

![Live-validated results across batch sizes -- reward per algorithm, and the energy-band mechanism at bs=16](RL/algos/results/results_overview.png)

### DDPG -- never produced a policy to validate

The original design (`RL/rl.py`) was a DDPG-style actor-critic: a Fast Actor
emitting three sigmoid-scaled continuous scalars, `torch.round`ed into the
three speculative-decoding integers, with a Target Actor for soft-update
stability and a Q-Critic scoring `[state, action]` pairs. There's no live
validation table for it because there was never a trained policy to validate:

| algorithm | bs=1 | bs=4 | bs=8 | bs=16 | avg reward | status |
|---|---|---|---|---|---|---|
| ddpg | -- | -- | -- | -- | -- | never trained |

Two independent, structural reasons, not a tuning issue:

1. `round()` has zero gradient almost everywhere, so backprop through the
   actor produced `grad: tensor([[0.]])` -- the actor never updated.
2. `profiler(eagle_3_sd)(...)` discarded its return value, so the reward it
   trained against was a hardcoded constant `0.9396` regardless of the action
   actually taken -- even a fixed gradient would have had nothing real to
   learn from.

Separately, only 75.4% of the continuous box
(`steps ∈ [1,32] × topk ∈ [1,10] × draft ∈ [1,64]`) satisfies sglang's own
validity constraints, so a quarter of the actions the actor could propose
were unrunnable even in principle. `RL/rl.py` is kept only as a reference for
what didn't work; it is not run to produce results.

### MLP contextual bandit -- current design, live A/B validated

Enumerates the legal triples into the fixed 19-action `ActionSpace` above and
trains a `QNetwork` to emit one Q-value per action; illegal actions are
masked to `-inf`, greedy selection is a single argmax. One-step contextual
bandit (`gamma=0`), not a sequential MDP -- the sweep captures no state
transition, so there's nothing for a discount factor to do.

Validated live 4 separate times: once with the trained policy, then three
more where the policy was *retrained from scratch with a different seed* and
re-evaluated fresh. All three seeds converged on the identical pick,
`(steps=3, topk=2, draft=4)`, at `bs=16` (the only bs where it diverges from
the reference at all):

| seed | ref reward | policy reward | delta |
|---|---|---|---|
| 0 | 0.3268 | 0.3869 | +0.0601 |
| 1 | 0.3181 | 0.3863 | +0.0682 |
| 2 | 0.0911 | 0.3852 | +0.2941 |
| 3 | 0.2978 | 0.4009 | +0.1032 |

reference: mean **0.2585**, std **0.097**  |  policy: mean **0.3898**, std
**0.0064**  |  mean delta **+0.131**

The mechanism is visible directly in telemetry, not inferred: the reference
config's energy utilization sat at **0.997, 1.005, 1.007** across the three
seed trials -- consistently *over* the 0.95-0.98 band and worsening as the
GPU heated up run to run, which is exactly what the reward function's 10x
overshoot penalty punishes. The policy's pick drew less power and landed at
**0.966, 0.974, 0.979** every time -- inside the band with margin, hence
higher reward on average *and* 15x more stable.

Saved model: `RL/policy.pth`, also published at
[`Pradheep1647/eagle3-speculative-decoding-policy`](https://huggingface.co/Pradheep1647/eagle3-speculative-decoding-policy).

### Seven alternative algorithms -- live validated, one trial, paired

`RL/algos/` builds and live-validates every other algorithm that's actually
applicable to this problem shape (small discrete action space, one reward per
pull, no captured state transition). Same live A/B methodology as the MLP
bandit above -- real sglang servers, launched right now on the physical GPU,
baseline and reference re-measured fresh per batch size so the comparison
shares the same thermal state:

| algorithm | bs=1 | bs=4 | bs=8 | bs=16 | avg reward | vs reference |
|---|---|---|---|---|---|---|
| *(reference, fixed)* | 0.5323 | 0.4975 | 0.4653 | 0.2945 | 0.4474 | -- |
| lookup_table | 0.5323 | 0.4975 | 0.4653 | 0.3855 | **0.4702** | **+0.0228** |
| doubly_robust | 0.5323 | 0.4975 | 0.4653 | 0.3855 | **0.4702** | **+0.0228** |
| linucb | 0.5323 | 0.4975 | 0.4113 | 0.3855 | 0.4567 | +0.0093 |
| gbt | 0.5173 | 0.4290 | 0.4653 | 0.2555 | 0.4168 | -0.0307 |
| thompson_sampling | 0.5173 | 0.4177 | 0.3413 | 0.2971 | 0.3934 | -0.0541 |
| bcq | 0.5323 | 0.0000 | 0.0000 | 0.0000 | 0.1331 | -0.3143 |
| cql | 0.4483 | 0.0000 | 0.0000 | 0.0000 | 0.1121 | -0.3353 |

Notes:

- **lookup_table, doubly_robust, and linucb all independently rediscover the
  MLP bandit's exact `bs=16` pick**, `(3,2,4)`, at a small fraction of the
  training cost (sub-second vs. seconds of MLP training) -- strong evidence
  the finding is a real property of the data, not an artifact of using a
  neural net.
- **cql and bcq collapsed onto the trivial non-speculative baseline `(0,0,0)`
  at `bs=4,8,16`.** Their `bs=4/8/16` reward is 0.0 by construction, not
  measurement -- `throughput_score` of a config against itself is always
  exactly 0, so no live launch was even needed to know the score. Root cause
  is hyperparameter default, not a bug: CQL's conservative penalty
  (`alpha=1.0`) and BCQ's imitation threshold (`threshold=0.3`) were left at
  their textbook defaults, which are far too aggressive relative to this
  dataset's reward scale (max ~0.53) -- both algorithms played it safe and
  picked "do nothing." Reported as-is rather than retuned.
- Discrete SAC/DQN were not run. Both only make sense once configs can be
  switched *mid-session* in response to evolving thermal state -- i.e. once
  there's an actual captured state transition and `gamma` stops being 0. The
  sweep here is one-shot per config, so training either would just be
  fitted-Q with extra steps and no real bootstrap target. Would need a
  genuinely different data collection pass first: a live session that keeps
  one server up and hot-swaps speculative params between requests, logging
  `(state, action, reward, next_state)` transitions instead of independent
  rows.

Saved models, one file per algorithm, published together at
[`Pradheep1647/eagle3-speculative-decoding-policy`](https://huggingface.co/Pradheep1647/eagle3-speculative-decoding-policy):

| algorithm | file |
|---|---|
| mlp_bandit | `mlp_bandit/policy.pth` |
| lookup_table | `lookup_table/model.json` |
| linucb | `linucb/model.npz` |
| thompson_sampling | `thompson_sampling/model.npz` |
| gbt | `gbt/model.joblib` |
| doubly_robust | `doubly_robust/model.joblib` |
| cql | `cql/policy.pth` |
| bcq | `bcq/policy.pth` |

Raw per-config live results: `RL/algos/results/live_validate.jsonl` /
`live_validate.log`. Per-algorithm picks: `RL/algos/results/picks.json`.

Would love to contribute and get the correct guidance.
