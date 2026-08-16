import argparse
import json

# common does the sys.path insert for the parent dir, has to import before the rest
from common import PICKS_PATH
from reward import compute_reward
from run_sweep import run_config
from sweep_config import BASELINE_CONFIG, REFERENCE_CONFIG

# same live a/b approach as eval_live.py -- launch real servers, benchmark
# baseline/reference/each algo's actual pick back to back on this gpu, right
# now. no reusing historical rows, paired within the same trial so thermal
# state is shared across everything compared at a given bs.
#
# only benchmarks bs where at least one algo actually diverged from the fixed
# reference -- if every algo just reproduces (3,4,8) at some bs there's
# nothing new to measure there, same reasoning that scoped eval_live.py down
# to bs=16 for the mlp bandit.


def load_picks():
    with open(PICKS_PATH) as f:
        return json.load(f)


def configs_to_run(picks, bs):
    # dedupe: (0,0,0) picks collapse into the baseline measurement, no separate
    # launch needed since it's the literal same config
    by_config = {}
    for algo, per_bs in picks.items():
        cfg = tuple(per_bs.get(str(bs), []))
        if len(cfg) != 3 or cfg == REFERENCE_CONFIG or cfg == BASELINE_CONFIG:
            continue
        by_config.setdefault(cfg, []).append(algo)
    return by_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--draft", default="rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct")
    p.add_argument("--output", default="results/live_validate.jsonl")
    p.add_argument("--port", type=int, default=30001)
    p.add_argument("--mem-fraction", type=float, default=0.55)
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--launch-timeout", type=int, default=900)
    p.add_argument("--power-limit", type=float, default=80.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    args = p.parse_args()

    picks = load_picks()
    batch_sizes = sorted({int(bs) for per_bs in picks.values() for bs in per_bs})

    plan = {bs: configs_to_run(picks, bs) for bs in batch_sizes}
    plan = {bs: cfgs for bs, cfgs in plan.items() if cfgs}  # skip bs with nothing new to test
    total_launches = sum(len(cfgs) + 2 for cfgs in plan.values())  # +2 per bs = baseline + reference
    print(f"plan: {len(plan)} batch sizes need live runs, {total_launches} total server launches")
    for bs, cfgs in plan.items():
        print(f"  bs={bs}: {list(cfgs.items())}")

    try:
        from components.profiler_cpu_gpu import HardwareMetricsProfiler
        profiler_instance = HardwareMetricsProfiler(gpu_index=0)
    except Exception as exc:
        print(f"profiler unavailable, continuing without energy metrics: {exc}")
        profiler_instance = None

    results = {}  # bs -> tag -> record
    for bs, cfgs in plan.items():
        results[bs] = {}
        run_list = [("baseline", BASELINE_CONFIG), ("reference", REFERENCE_CONFIG)]
        run_list += [(f"cfg_{cfg[0]}_{cfg[1]}_{cfg[2]}", cfg) for cfg in cfgs]

        for tag, cfg3 in run_list:
            cfg = (bs,) + cfg3
            print(f"[{tag}] bs={bs} steps={cfg3[0]} topk={cfg3[1]} draft={cfg3[2]} ...", flush=True)
            record = run_config(cfg, args, profiler_instance)
            record["tag"] = tag
            with open(args.output, "a") as fout:
                fout.write(json.dumps(record) + "\n")
            results[bs][tag] = record
            if "error" in record:
                print(f"    ERROR {record['error']}", flush=True)
            else:
                print(f"    accept={record.get('accept_length')} speed={record.get('speed_tok_s')} "
                      f"tok/s J/tok={record.get('joules_per_token')}", flush=True)

    print("\n=== live validation: every algo's actual pick vs fixed reference, measured just now ===")
    for bs, cfgs in plan.items():
        base = results[bs].get("baseline")
        ref = results[bs].get("reference")
        base_speed = base.get("speed_tok_s") if base and not base.get("error") else None

        ref_reward = None
        if ref and not ref.get("error"):
            ref_reward, _ = compute_reward(ref, base_speed, args.power_limit, args.energy_weight)

        print(f"\nbs={bs}")
        if ref_reward is not None:
            print(f"  {'reference':<12} {str(REFERENCE_CONFIG):<12} reward={ref_reward:.4f} "
                  f"speed={ref.get('speed_tok_s')}")
        for cfg, algos in cfgs.items():
            tag = f"cfg_{cfg[0]}_{cfg[1]}_{cfg[2]}"
            row = results[bs].get(tag)
            if row is None or row.get("error"):
                print(f"  {str(cfg):<12} picked_by={algos}  ERROR {row.get('error') if row else 'missing'}")
                continue
            rwd, parts = compute_reward(row, base_speed, args.power_limit, args.energy_weight)
            delta = rwd - ref_reward if ref_reward is not None else None
            delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
            print(f"  {str(cfg):<12} picked_by={algos}")
            print(f"    reward={rwd:.4f} (vs reference {delta_str})  speed={row.get('speed_tok_s')} "
                  f"util={parts.get('energy_utilization') and round(parts['energy_utilization'], 3)}")


if __name__ == "__main__":
    main()
