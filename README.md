# Energy-Aware Speculative Decoding Control

An RL framework that dynamically picks EAGLE3 speculative-decoding parameters
(`speculative_num_steps`, `speculative_eagle_topk`, `speculative_num_draft_tokens`)
to hold GPU energy utilization (average power draw / power limit) inside a
tight 95-98% band, served through sglang. Below the band the GPU has idle
capacity the policy should spend on deeper speculation; above it the card
approaches its power cap, clocks down, and more speculation buys nothing but
costs energy.

## Architecture: contextual bandit, not DDPG

The original design (see `RL/rl.py`) was a DDPG-style actor-critic: a Fast
Actor emitted three sigmoid-scaled continuous scalars, `torch.round`ed into
the three speculative-decoding integers, with a Target Actor for soft-update
stability and a Q-Critic scoring `[state, action]` pairs. It didn't work, for
two structural reasons (see `RL/policy.py`):

1. `round()` has zero gradient almost everywhere, so backprop through the
   actor produced `grad: tensor([[0.]])` -- the actor never learned.
2. Only 75.4% of the continuous box (`steps∈[1,32] × topk∈[1,10] × draft∈[1,64]`)
   satisfies sglang's own validity constraints (`RL/sweep_config.py`), so a
   quarter of proposed actions were unrunnable even with a working gradient.

The current design (`RL/policy.py`, `RL/train_offline.py`) enumerates the
legal `(steps, topk, num_draft_tokens)` triples into a fixed discrete
`ActionSpace` (19 actions) and trains a `QNetwork` to emit one Q-value per
action; illegal actions are masked to `-inf` and greedy selection is a single
argmax. Because the training data is a cached sweep -- one reward per
`(batch_size, config)` with no captured state transition, nothing carries
over from one config to the next -- this is a one-step contextual bandit
(`gamma = 0`), not a sequential MDP. Calling it DDPG would be dressing up a
bandit.

Data collection (`RL/run_sweep.py`) launches a real sglang server per config
on an RTX 4060 Laptop GPU (8GB) and benchmarks it directly -- no server is
ever launched inside the training loop, so training itself runs free on CPU
in a few seconds. Dataset: 228 rows, 3 full repeats of a 76-cell grid
(4 batch sizes × 19 actions), published at
[`Pradheep1647/eagle3-speculative-decoding-energy-sweep`](https://huggingface.co/datasets/Pradheep1647/eagle3-speculative-decoding-energy-sweep).

## Findings

**Validated result at `batch_size=16`: the learned policy beats the fixed
reference config `(steps=3, topk=4, draft=8)` by staying inside the energy
band instead of overshooting it.**

This isn't just an offline metric on stored rows -- it was checked two
separate ways:

- **Leave-one-repeat-out holdout** (`RL/validate_holdout.py`): trained on
  repeats 0+1, evaluated against repeat 2 (never seen during training). The
  policy's pick scored 0.383 vs. the reference's 0.305 on data the network
  never trained on. (`batch_size=1,4,8` either matched the reference trivially
  or -- in the case of `bs=8` -- looked like a win in-sample but did not
  survive the holdout split, i.e. it was overfitting noise, not a real
  effect.)
- **Live A/B, 4 independent trials** (`RL/eval_live.py`, `RL/eval_live_seeds.py`):
  actually launching real sglang servers and re-benchmarking, right now, on
  the physical GPU -- one run with the already-trained policy, plus three
  more where the policy was *retrained from scratch with a different seed
  each time* and re-evaluated with a fresh forward pass. All three seeds
  converged to the identical pick, `(steps=3, topk=2, draft=4)`:

  | seed | ref reward | policy reward | delta |
  |---|---|---|---|
  | 0 | 0.3268 | 0.3869 | +0.0601 |
  | 1 | 0.3181 | 0.3863 | +0.0682 |
  | 2 | 0.0911 | 0.3852 | +0.2941 |
  | 3 | 0.2978 | 0.4009 | +0.1032 |

  reference: mean **0.2585**, std **0.097** &nbsp;&nbsp;|&nbsp;&nbsp;
  policy: mean **0.3898**, std **0.0064** &nbsp;&nbsp;|&nbsp;&nbsp;
  mean delta **+0.131**

  The mechanism is visible directly in the telemetry, not inferred: the
  reference config's energy utilization was **0.997, 1.005, 1.007** across
  the three seed trials -- consistently *over* the 0.95-0.98 band, and
  worsening as the GPU heated up run to run. The reward function's overshoot
  penalty (10x slope) punishes that hard and inconsistently depending on
  exactly how far over it drifts, which is why the reference's reward is
  volatile (std 0.097). The policy's pick draws less power and landed at
  **0.966, 0.974, 0.979** every time -- inside the band with margin, which is
  why its reward is both higher on average *and* 15x more stable.

  At `batch_size=1,4,8` the trained policy just reproduces the reference
  config exactly -- it only diverges (and only needs to) at `bs=16`, where the
  reference's power draw is high enough to overshoot the band.

Caveat worth flagging: `state_features()` conditions on `gpu_mem_used_mb` and
`gpu_util_pct` as recorded *after* a config already ran, not before it was
chosen -- the sweep has no true pre-decision observation for those two. All
evaluation so far (in-sample, holdout, and live) inherits that, so "the
policy's pick" means "what the network outputs given a post-hoc state", not
a live sensor read taken before the config was selected. This is fine for
picking a static per-batch-size config (what's tested above) but would need
fixing before trusting the policy to react to genuinely live GPU state
mid-session.

## Algorithms worth testing besides DDPG

DDPG was the wrong tool because the action space is small and discrete, not
continuous, and the problem has no captured state transition. Given that,
here's what's actually applicable, roughly in order of how well they fit:

- **Contextual bandit baselines (LinUCB, Thompson Sampling).** The problem
  *is* a contextual bandit -- 19 discrete actions, a 4-dim context, one
  reward per pull. LinUCB or a Bayesian linear/Thompson-sampling bandit would
  fit the structure exactly and, unlike the current fitted-Q approach, comes
  with a principled exploration strategy for the case where you're allowed to
  explore live instead of training purely offline on a fixed sweep.
- **Off-policy bandit evaluation (IPS / doubly robust).** The sweep's
  logging policy is uniform-random over a known grid -- about as clean as a
  behavior policy gets. Rather than fitting `Q(s,a)` by regression (what
  `train_offline.py` does now), inverse-propensity-weighted or doubly-robust
  estimators could directly optimize an off-policy value estimate, which is
  the statistically principled way to learn from fixed logged data in a
  bandit setting.
- **Conservative / offline RL (CQL, BCQ).** With only 3 samples per action,
  there's real epistemic uncertainty the current network doesn't account for
  -- it happily extrapolates a Q-value for any action regardless of how
  little data supports it. CQL-style penalties on out-of-distribution
  actions would guard against confidently picking a poorly-sampled config.
- **Non-neural regression baselines.** 225 rows and a 4-dim context is small
  for a 2×128-hidden MLP. Gradient-boosted trees (or even a per-batch-size
  lookup table of the empirically best config) are worth running side by
  side -- partly as a sanity check on whether the network is earning its
  complexity, and partly because tree-based feature importance would give a
  direct, quantitative answer to "does this policy actually use
  temp/mem/util, or is it just keying off batch_size" instead of the
  qualitative read we have now.
- **Discrete SAC / DQN -- only if the framing changes.** These make sense
  once configs can be switched *mid-session* in response to evolving thermal
  state, i.e. once there's an actual captured state transition and `gamma`
  stops being 0. Discrete SAC in particular handles a discrete action space
  natively, without the `round()`-gradient failure that sank the original
  DDPG design. Not worth it under the current one-shot-per-request framing.

Would love to contribute and get the correct guidance.
