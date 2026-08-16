"""Leave-one-repeat-out validation.

train_offline.py's evaluate() is in-sample: it trains on every row, then checks
what the policy picks using a state drawn from that same training data. This
script instead trains only on repeats 0+1 and scores the resulting policy's
choices against repeat 2 -- a thermal-noise realization of the same grid the
network never saw during training.

Usage:
  python3 validate_holdout.py --dataset hf_push/eagle3_energy_sweep.jsonl --power-limit 80
"""

import argparse
import random

import numpy as np
import torch

from policy import ActionSpace, QNetwork
from reward import compute_reward
from sweep_config import REFERENCE_CONFIG
from train_offline import build_dataset, load_rows, state_features, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hf_push/eagle3_energy_sweep.jsonl")
    p.add_argument("--power-limit", type=float, default=80.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = load_rows(args.dataset)
    train_rows = [r for r in rows if r.get("repeat") in (0, 1)]
    test_rows = [r for r in rows if r.get("repeat") == 2]

    space, states, actions, rewards, meta, baselines = build_dataset(
        train_rows, args.power_limit, args.energy_weight)
    print(f"train: {len(train_rows)} rows -> {len(states)} usable (repeats 0+1)")
    print(f"test:  {len(test_rows)} rows (repeat 2, held out)")

    q = QNetwork(state_dims=states.shape[1], n_actions=len(space))
    train(q, states, actions, rewards, args.epochs, args.batch_size, args.seed)

    test_baselines = {r["batch_size"]: r.get("speed_tok_s")
                       for r in test_rows if r.get("is_baseline") and not r.get("error")}

    by_key = {}
    for r in test_rows:
        if r.get("error"):
            continue
        rwd, _ = compute_reward(r, test_baselines.get(r["batch_size"]),
                                 args.power_limit, args.energy_weight)
        by_key[(r["batch_size"], r["steps"], r["topk"], r["num_draft_tokens"])] = (rwd, r)

    print("\n=== held-out repeat=2 evaluation (network never trained on these rows) ===")
    for bs in sorted(test_baselines):
        bs_rows = [r for r in test_rows if r["batch_size"] == bs and not r.get("error")]
        if not bs_rows:
            continue
        state = torch.tensor([state_features(bs_rows[0])], dtype=torch.float).to(q.device)
        chosen = space.actions[q.act(state, space.mask(), epsilon=0.0)]

        chosen_result = by_key.get((bs,) + chosen)
        ref_result = by_key.get((bs,) + REFERENCE_CONFIG)
        bs_results = [(k, v) for k, v in by_key.items() if k[0] == bs]
        oracle_key, oracle_result = max(bs_results, key=lambda kv: kv[1][0])

        def fmt(tag, cfg, result):
            if result is None:
                return f"  {tag:<12} {str(cfg):<12} (not in held-out data)"
            rwd, row = result
            return f"  {tag:<12} {str(cfg):<12} reward={rwd:.4f} speed={row.get('speed_tok_s')}"

        print(f"bs={bs}")
        print(fmt("policy", chosen, chosen_result))
        print(fmt("reference", REFERENCE_CONFIG, ref_result))
        print(fmt("oracle", oracle_key[1:], oracle_result))


if __name__ == "__main__":
    main()
