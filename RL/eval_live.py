"""Live A/B: run the policy's actual picks on a real sglang server, right now.

Everything else in this repo either trains on the cached sweep or re-scores
already-collected rows against it. This script does neither: it asks the
trained policy which config it would deploy for each batch size, then
launches real servers and benchmarks baseline / reference / policy back to
back on the current machine, so the comparison is fresh measurement against
fresh measurement, not fresh-vs-historical.

Caveat: the policy's state features (state_features() in train_offline.py)
include gpu_mem_used_mb and gpu_util_pct as recorded *after* a config ran,
not before it was chosen -- the training data has no true pre-decision
observation for those two. This script inherits that (it derives the
decision state from a representative dataset row, same as evaluate() and
validate_holdout.py already do), so "the policy's pick" here still means
"what the network outputs given a post-hoc state", not a live sensor read
taken before the config was selected.

Usage:
  python3 eval_live.py --target unsloth/Llama-3.2-1B-Instruct \
      --draft rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct \
      --mem-fraction 0.55
"""

import argparse
import json

import torch

from policy import ActionSpace, QNetwork
from reward import compute_reward
from run_sweep import run_config
from sweep_config import BASELINE_CONFIG, REFERENCE_CONFIG
from train_offline import load_rows, state_features


def pick_policy_configs(dataset_path, policy_path, batch_sizes):
    rows = load_rows(dataset_path)
    space = ActionSpace()
    ckpt = torch.load(policy_path, map_location="cpu")
    q = QNetwork(state_dims=4, n_actions=len(space))
    q.load_state_dict(ckpt["state_dict"])
    q.eval()

    chosen = {}
    for bs in batch_sizes:
        bs_rows = [r for r in rows if r["batch_size"] == bs and not r.get("error")]
        if not bs_rows:
            raise SystemExit(f"no data for batch_size={bs} to derive a state from")
        state = torch.tensor([state_features(bs_rows[0])], dtype=torch.float).to(q.device)
        action_idx = q.act(state, space.mask(), epsilon=0.0)
        chosen[bs] = space.actions[action_idx]
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--draft", default="rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct")
    p.add_argument("--dataset", default="hf_push/eagle3_energy_sweep.jsonl")
    p.add_argument("--policy", default="policy.pth")
    p.add_argument("--output", default="live_eval.jsonl")
    p.add_argument("--port", type=int, default=30001)
    p.add_argument("--mem-fraction", type=float, default=0.55)
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--launch-timeout", type=int, default=900)
    p.add_argument("--power-limit", type=float, default=80.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    args = p.parse_args()

    chosen = pick_policy_configs(args.dataset, args.policy, args.batch_sizes)
    print("policy picks:", chosen, flush=True)

    try:
        from components.profiler_cpu_gpu import HardwareMetricsProfiler
        profiler_instance = HardwareMetricsProfiler(gpu_index=0)
    except Exception as exc:
        print(f"profiler unavailable, continuing without energy metrics: {exc}")
        profiler_instance = None

    results = []
    for bs in args.batch_sizes:
        for tag, cfg3 in [("baseline", BASELINE_CONFIG), ("reference", REFERENCE_CONFIG),
                           ("policy", chosen[bs])]:
            cfg = (bs,) + cfg3
            print(f"[{tag}] bs={bs} steps={cfg3[0]} topk={cfg3[1]} draft={cfg3[2]} ...", flush=True)
            record = run_config(cfg, args, profiler_instance)
            record["tag"] = tag
            with open(args.output, "a") as fout:
                fout.write(json.dumps(record) + "\n")
            results.append(record)
            if "error" in record:
                print(f"    ERROR {record['error']}", flush=True)
            else:
                print(f"    accept={record.get('accept_length')} speed={record.get('speed_tok_s')} "
                      f"tok/s J/tok={record.get('joules_per_token')}", flush=True)

    print("\n=== live A/B: policy pick vs fixed reference (measured just now, paired) ===")
    for bs in args.batch_sizes:
        base = next((r for r in results if r["batch_size"] == bs and r["tag"] == "baseline"), None)
        ref = next((r for r in results if r["batch_size"] == bs and r["tag"] == "reference"), None)
        pol = next((r for r in results if r["batch_size"] == bs and r["tag"] == "policy"), None)
        base_speed = base.get("speed_tok_s") if base and not base.get("error") else None

        print(f"bs={bs}  policy_config={chosen[bs]}")
        for tag, row in [("reference", ref), ("policy", pol)]:
            if row is None or row.get("error"):
                print(f"  {tag:<10} ERROR {row.get('error') if row else 'missing'}")
                continue
            rwd, parts = compute_reward(row, base_speed, args.power_limit, args.energy_weight)
            print(f"  {tag:<10} reward={rwd:.4f} speed={row.get('speed_tok_s')} "
                  f"util={parts.get('energy_utilization') and round(parts['energy_utilization'], 3)} "
                  f"J/tok={row.get('joules_per_token')}")


if __name__ == "__main__":
    main()
