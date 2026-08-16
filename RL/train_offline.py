"""Train the policy on the cached sweep -- laptop only, no GPU server involved.

The original loop in rl.py launched an SGLang server inside every iteration
(~90 s x 10,000 iterations, all on metered hardware). Since the reward for a
config is a property of the config, not of the training step, it can be measured
once and reused: the A100 collects sweep.jsonl, and everything below runs free
on CPU.

Because the sweep captures no state transition -- each row is an independent
measurement, and nothing carries over from one config to the next -- this is a
contextual bandit. The Q-target is therefore just the immediate reward, with no
bootstrap term. See policy.py.

Usage:
  python3 train_offline.py --dataset sweep.jsonl --power-limit 400
"""

import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from policy import ActionSpace, QNetwork
from reward import compute_reward
from sweep_config import BASELINE_CONFIG, REFERENCE_CONFIG


def load_rows(path):
    rows = []
    with open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def state_features(row):
    """Context the policy conditions on.

    gpu_fan_speed_percent, battery_percent and battery_plugged_in from
    get_normalized_state_vector are omitted: the A100-SXM4 is passively cooled
    (fan speed reads as unsupported) and a datacenter node has no battery. They
    were constant across every row, so they carry no signal.
    """
    return [
        row["batch_size"] / 8.0,
        (row.get("gpu_temp_c_before") or 40.0) / 100.0,
        (row.get("gpu_mem_used_mb") or 0.0) / 81920.0,
        (row.get("gpu_util_pct") or 0.0) / 100.0,
    ]


def build_dataset(rows, power_limit, energy_weight):
    baselines = {r["batch_size"]: r.get("speed_tok_s")
                 for r in rows if r.get("is_baseline") and not r.get("error")}
    if not baselines:
        raise SystemExit("no baseline row in dataset -- rerun the sweep including (0,0,0)")

    space = ActionSpace()
    states, actions, rewards, meta = [], [], [], []
    for row in rows:
        if row.get("error"):
            continue
        idx = space.to_index(row["steps"], row["topk"], row["num_draft_tokens"])
        if idx is None:
            continue
        r, parts = compute_reward(row, baselines.get(row["batch_size"]), power_limit,
                                  energy_weight=energy_weight)
        states.append(state_features(row))
        actions.append(idx)
        rewards.append(r)
        meta.append({**row, **parts, "reward": r})

    return (space,
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            meta, baselines)


def train(q, states, actions, rewards, epochs, batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    n = len(states)
    states, actions, rewards = (states.to(q.device), actions.to(q.device), rewards.to(q.device))
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g).to(q.device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            pred = q(states[b]).gather(1, actions[b].unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(pred, rewards[b])
            q.optimizer.zero_grad()
            loss.backward()
            q.optimizer.step()
            epoch_loss += loss.item() * len(b)
        losses.append(epoch_loss / n)
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1:4d}  loss {losses[-1]:.5f}")
    return losses


def evaluate(q, space, meta, batch_sizes):
    """Compare the learned policy against the fixed reference config."""
    by_key = {(m["batch_size"], m["steps"], m["topk"], m["num_draft_tokens"]): m for m in meta}
    print("\n=== policy vs fixed reference ===")
    for bs in batch_sizes:
        rows = [m for m in meta if m["batch_size"] == bs]
        if not rows:
            continue
        state = torch.tensor([state_features(rows[0])], dtype=torch.float).to(q.device)
        chosen = space.actions[q.act(state, space.mask(), epsilon=0.0)]
        chosen_row = by_key.get((bs,) + chosen)
        ref_row = by_key.get((bs,) + REFERENCE_CONFIG)
        best = max(rows, key=lambda m: m["reward"])

        def fmt(tag, row, cfg):
            if row is None:
                return f"  {tag:<12} {str(cfg):<12} (not in dataset)"
            return (f"  {tag:<12} {str(cfg):<12} reward={row['reward']:.4f} "
                    f"speed={row.get('speed_tok_s')} util={row.get('energy_utilization') and round(row['energy_utilization'], 3)}")

        print(f"bs={bs}")
        print(fmt("policy", chosen_row, chosen))
        print(fmt("reference", ref_row, REFERENCE_CONFIG))
        print(fmt("oracle", best, (best["steps"], best["topk"], best["num_draft_tokens"])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sweep.jsonl")
    p.add_argument("--power-limit", type=float, default=400.0,
                   help="GPU power cap in watts (A100-SXM4-80GB = 400)")
    p.add_argument("--energy-weight", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = load_rows(args.dataset)
    space, states, actions, rewards, meta, baselines = build_dataset(
        rows, args.power_limit, args.energy_weight)
    print(f"{len(rows)} rows -> {len(states)} usable, {len(space)} discrete actions")
    print(f"baseline speeds: {baselines}")
    print(f"reward range: {rewards.min():.4f} .. {rewards.max():.4f}")

    q = QNetwork(state_dims=states.shape[1], n_actions=len(space))
    train(q, states, actions, rewards, args.epochs, args.batch_size, args.seed)
    evaluate(q, space, meta, sorted({m["batch_size"] for m in meta}))

    torch.save({"state_dict": q.state_dict(), "actions": space.actions}, "policy.pth")
    print("\nsaved policy.pth")


if __name__ == "__main__":
    main()
