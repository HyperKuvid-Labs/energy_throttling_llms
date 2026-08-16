# Energy-Aware Speculative Decoding Control

A cookbook, not a paper. The deliverable is a small table of
`(speculative_num_steps, speculative_eagle_topk, speculative_num_draft_tokens)`
values per batch size that are known-good for EAGLE3 + sglang on this GPU, so
the params don't get guessed by hand every time a server spins up. The sweep,
the reward, and the eight algorithms below all exist to earn that table.

Target: hold GPU energy utilization (avg power / power limit) inside a tight
**95-98% band**. Below it there's idle capacity worth spending on deeper
speculation; above it the card is near its power cap, clocks down, and more
speculation just costs energy for nothing.

[![🤗 Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-eagle3--speculative--decoding--energy--sweep-FFD21E)](https://huggingface.co/datasets/Pradheep1647/eagle3-speculative-decoding-energy-sweep)
[![🤗 Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-eagle3--speculative--decoding--policy-FFD21E)](https://huggingface.co/Pradheep1647/eagle3-speculative-decoding-policy)

## The recipe

Live-measured on an RTX 4060 Laptop GPU (8GB), `unsloth/Llama-3.2-1B-Instruct`
+ `rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct` draft head, 80W cap:

| batch_size | steps | topk | draft | live reward | vs fixed `(3,4,8)` reference |
|---|---|---|---|---|---|
| 1  | 3 | 4 | 8 | 0.5323 | same config |
| 4  | 3 | 4 | 8 | 0.4975 | same config |
| 8  | 3 | 4 | 8 | 0.4653 | same config |
| 16 | 3 | 2 | 4 | 0.3855 | **+0.091**, avoids overshooting the band |

`bs=16` is the only case worth changing the launch flags for -- and four
independent methods (MLP bandit, lookup table, LinUCB, doubly robust) all
converge on the same `(3,2,4)` pick.

## Model I/O

**Input** -- 4-dim state, read from GPU telemetry after a config runs (not a
live pre-decision sensor read, see caveat below):

```
[ batch_size / 8.0, gpu_temp_c_before / 100.0, gpu_mem_used_mb / 8192.0, gpu_util_pct / 100.0 ]
```

**Output** -- an index into a fixed `ActionSpace` (`RL/policy.py`), decoded to
`--speculative-num-steps`, `--speculative-eagle-topk`,
`--speculative-num-draft-tokens`.

Grid before filtering: `steps ∈ {0,1,3,5} × topk ∈ {0,1,2,4} × draft ∈
{0,2,4,8,16}`, across `batch_size ∈ {1,4,8,16}` (batch size is a state
feature, not part of the action). Validity constraints, from sglang 0.5.2
source:

1. `steps * topk + 1 >= num_draft_tokens` -- surplus draft tokens can never
   be filled.
2. `topk == 1` forces `num_draft_tokens = steps + 1` server-side, collapsing
   two nominal configs into one.
3. `(0, 0, 0)` is the non-speculative baseline; any other zero-containing
   combo is invalid.

After validity + dedup: **19 distinct actions**.

Caveat: `gpu_mem_used_mb` / `gpu_util_pct` come from `metrics_after`, i.e.
*after* that config already ran, not a genuine pre-decision read. Fine for
picking a static per-batch-size config (everything here); would need fixing
before trusting live mid-session reactions.

## The sweep

`RL/run_sweep.py` launches a real sglang server per `(batch_size, config)` on
the physical GPU. The sweep runs once, caches to JSONL, and every algorithm
below trains against that cache, free, on CPU.

Per config: launch sglang (target + draft both forced to `dtype=float16`,
since EAGLE3's draft head ships fp16 and the target ships bf16 -- a mismatch
kills CUDA graph capture) → poll `/health` → warmup batch → snapshot energy
(`nvmlDeviceGetTotalEnergyConsumption`) and telemetry → fire
`max(num_prompts, batch_size)` requests at concurrency = batch_size against 4
fixed prompts, 512 output tokens each (same methodology as sglang's own
`bench_speculative.py`) → snapshot again, read `/get_server_info` for
`avg_spec_accept_length` and `step_time_dict` → `speed_tok_s =
accept_length / p20(step_time)` → tear down, sleep 5s to let the GPU settle.

Reward (`RL/reward.py`), computed offline from the cached row:

- `energy_utilization = avg_power_watts / power_limit_watts` (80W cap)
- `band_score`: 1.0 inside `[0.95, 0.98]`, linear ramp below, 10x-slope falloff above
- `throughput_score = clip((speed / baseline_speed - 1) / 2, 0, 1)` -- a
  config scored against itself is always exactly 0 here, by construction
- `reward = (band_score^w * throughput_score^(1-w)) * thermal_multiplier`
  (geometric mean, `w=0.5`), then a penalty for `>75°C` (0.8x), `>80°C`
  (0.5x), or active throttling (0.3x). Clipped to `[0, 1]`.

**Dataset**: 228 rows, 3 repeats of the 76-cell grid (4 batch sizes x 19
actions), landing at different thermal states per repeat. One config
(`bs=16, steps=5, topk=4, draft=16`) never fits in 8GB and is left as an
error row. Full details and schema on the [dataset card](https://huggingface.co/datasets/Pradheep1647/eagle3-speculative-decoding-energy-sweep).

## Results

![Live-validated results across batch sizes -- reward per algorithm, and the energy-band mechanism at bs=16](RL/algos/results/results_overview.png)

### DDPG -- never produced a policy to validate

The original design (`RL/rl.py`): a Fast Actor emitting three sigmoid-scaled
continuous scalars, `torch.round`ed into the three integers, a Target Actor
for soft-update stability, a Q-Critic on `[state, action]`. No live
validation table -- there was never a trained policy:

| algorithm | bs=1 | bs=4 | bs=8 | bs=16 | avg reward | status |
|---|---|---|---|---|---|---|
| ddpg | -- | -- | -- | -- | -- | never trained |

Two independent, structural reasons:

1. `round()` has zero gradient almost everywhere -- backprop through the
   actor produced `grad: tensor([[0.]])`, so it never updated.
2. `profiler(eagle_3_sd)(...)` discarded its return value, so the reward it
   trained against was a hardcoded constant `0.9396` regardless of action.

Separately, only 75.4% of the continuous box (`steps ∈ [1,32] × topk ∈
[1,10] × draft ∈ [1,64]`) satisfies sglang's own validity constraints.
`RL/rl.py` is kept as a reference for what didn't work, not run for results.

### MLP contextual bandit -- current design, live A/B validated

Enumerates the legal triples into the 19-action `ActionSpace`, trains a
`QNetwork` for one Q-value per action, masks illegal actions to `-inf`,
picks greedily. One-step contextual bandit (`gamma=0`) -- the sweep captures
no state transition, so there's nothing for a discount factor to do.

Validated live 4 times: once with the trained policy, three more retrained
from scratch with a different seed each time. All converged on
`(steps=3, topk=2, draft=4)` at `bs=16`, the only bs where it diverges from
the reference:

| seed | ref reward | policy reward | delta |
|---|---|---|---|
| 0 | 0.3268 | 0.3869 | +0.0601 |
| 1 | 0.3181 | 0.3863 | +0.0682 |
| 2 | 0.0911 | 0.3852 | +0.2941 |
| 3 | 0.2978 | 0.4009 | +0.1032 |

reference: mean **0.2585**, std **0.097**  |  policy: mean **0.3898**, std
**0.0064**  |  mean delta **+0.131**

Visible directly in telemetry: the reference config's energy utilization sat
at **0.997, 1.005, 1.007** across the three seed trials -- consistently over
the band, worsening as the GPU heated up. The policy's pick landed at
**0.966, 0.974, 0.979** every time -- inside the band, higher reward on
average *and* 15x more stable.

### Seven alternative algorithms -- live validated, one paired trial

`RL/algos/` builds and live-validates every other algorithm applicable to
this problem shape (small discrete action space, one reward per pull, no
captured state transition). Same live A/B methodology, baseline and
reference re-measured fresh per batch size:

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

- **lookup_table, doubly_robust, and linucb all independently rediscover the
  MLP bandit's exact `bs=16` pick**, `(3,2,4)`, at a fraction of the training
  cost -- strong evidence it's a real property of the data.
- **cql and bcq collapsed onto the trivial non-speculative baseline
  `(0,0,0)`** at `bs=4,8,16` (reward 0 by construction, not measurement).
  Root cause is default hyperparameters far too conservative for this
  reward scale (CQL's `alpha=1.0`, BCQ's `threshold=0.3`), not a bug.
  Reported as-is rather than retuned.
- Discrete SAC/DQN weren't run -- they only make sense once configs switch
  *mid-session* against evolving thermal state (`gamma != 0`), which needs a
  genuinely different data collection pass: a live session hot-swapping
  speculative params and logging real `(state, action, reward, next_state)`
  transitions instead of independent rows.

**Saved models**, one file per algorithm, all in the [models repo](https://huggingface.co/Pradheep1647/eagle3-speculative-decoding-policy):

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
