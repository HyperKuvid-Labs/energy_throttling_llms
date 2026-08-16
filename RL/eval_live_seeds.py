"""Live A/B across training seeds: retrain the policy from scratch per seed,
take its greedy bs=16 pick via a forward pass, and benchmark that pick live --
paired against a freshly-measured baseline and reference in the same trial.

This tests whether the bs=16 result (policy beats the fixed reference by
staying in the energy-utilization band) holds across training-seed
variation, not just GPU-measurement noise on a single frozen policy.pth
(that's what eval_live.py already checked).

Usage:
  python3 eval_live_seeds.py --seeds 1 2 3 --batch-size 16 --mem-fraction 0.55
"""

import argparse
import json
import random

import numpy as np
import torch

from policy import ActionSpace, QNetwork
from reward import compute_reward
from run_sweep import run_config
from sweep_config import BASELINE_CONFIG, REFERENCE_CONFIG
from train_offline import build_dataset, load_rows, state_features, train


def train_and_pick(rows, power_limit, energy_weight, epochs, train_batch_size, seed, target_bs):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    space, states, actions, rewards, meta, baselines = build_dataset(rows, power_limit, energy_weight)
    q = QNetwork(state_dims=states.shape[1], n_actions=len(space))
    train(q, states, actions, rewards, epochs, train_batch_size, seed)

    bs_rows = [r for r in rows if r["batch_size"] == target_bs and not r.get("error")]
    state = torch.tensor([state_features(bs_rows[0])], dtype=torch.float).to(q.device)
    action_idx = q.act(state, space.mask(), epsilon=0.0)
    return space.actions[action_idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--draft", default="rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct")
    p.add_argument("--dataset", default="hf_push/eagle3_energy_sweep.jsonl")
    p.add_argument("--output", default="live_eval_seeds.jsonl")
    p.add_argument("--port", type=int, default=30001)
    p.add_argument("--mem-fraction", type=float, default=0.55)
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--launch-timeout", type=int, default=900)
    p.add_argument("--power-limit", type=float, default=80.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--train-batch-size", type=int, default=8,
                    help="training minibatch size, unrelated to --batch-size")
    p.add_argument("--batch-size", type=int, default=16, help="sweep batch_size to evaluate")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()

    rows = load_rows(args.dataset)

    try:
        from components.profiler_cpu_gpu import HardwareMetricsProfiler
        profiler_instance = HardwareMetricsProfiler(gpu_index=0)
    except Exception as exc:
        print(f"profiler unavailable, continuing without energy metrics: {exc}")
        profiler_instance = None

    trials = []
    for seed in args.seeds:
        chosen = train_and_pick(rows, args.power_limit, args.energy_weight, args.epochs,
                                 args.train_batch_size, seed, args.batch_size)
        print(f"\n--- seed={seed}  policy picks bs={args.batch_size} -> {chosen} ---", flush=True)

        trial_results = {}
        for tag, cfg3 in [("baseline", BASELINE_CONFIG), ("reference", REFERENCE_CONFIG), ("policy", chosen)]:
            cfg = (args.batch_size,) + cfg3
            print(f"[seed={seed}][{tag}] bs={args.batch_size} steps={cfg3[0]} topk={cfg3[1]} "
                  f"draft={cfg3[2]} ...", flush=True)
            record = run_config(cfg, args, profiler_instance)
            record["tag"], record["seed"] = tag, seed
            with open(args.output, "a") as fout:
                fout.write(json.dumps(record) + "\n")
            trial_results[tag] = record
            if "error" in record:
                print(f"    ERROR {record['error']}", flush=True)
            else:
                print(f"    accept={record.get('accept_length')} speed={record.get('speed_tok_s')} "
                      f"tok/s J/tok={record.get('joules_per_token')}", flush=True)

        base = trial_results["baseline"]
        base_speed = base.get("speed_tok_s") if not base.get("error") else None
        ref_reward = (compute_reward(trial_results["reference"], base_speed, args.power_limit,
                                      args.energy_weight)[0]
                      if not trial_results["reference"].get("error") else 0.0)
        pol_reward = (compute_reward(trial_results["policy"], base_speed, args.power_limit,
                                      args.energy_weight)[0]
                      if not trial_results["policy"].get("error") else 0.0)

        trials.append({"seed": seed, "policy_config": chosen,
                        "ref_reward": ref_reward, "pol_reward": pol_reward})

    print(f"\n=== summary across seeds (bs={args.batch_size}) ===")
    for t in trials:
        print(f"  seed={t['seed']}  policy_config={t['policy_config']}  "
              f"ref_reward={t['ref_reward']:.4f}  policy_reward={t['pol_reward']:.4f}  "
              f"delta={t['pol_reward'] - t['ref_reward']:+.4f}")

    ref_rewards = np.array([t["ref_reward"] for t in trials])
    pol_rewards = np.array([t["pol_reward"] for t in trials])
    print(f"\n  reference: mean={ref_rewards.mean():.4f} std={ref_rewards.std():.4f}")
    print(f"  policy:    mean={pol_rewards.mean():.4f} std={pol_rewards.std():.4f}")
    print(f"  mean delta (policy - reference): {(pol_rewards - ref_rewards).mean():+.4f}")


if __name__ == "__main__":
    main()
