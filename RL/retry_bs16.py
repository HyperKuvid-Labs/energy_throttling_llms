"""Rerun the configs that failed in laptop_sweep.jsonl for a given batch size,
at a lower --mem-fraction-static so the speculative-tree buffers fit.

Reuses run_config() from run_sweep.py so measurement is identical to the main
sweep; only the mem_fraction and the (scattered) list of configs to run differ.

Usage:
  python3 retry_bs16.py --source laptop_sweep.jsonl --output bs16_retry.jsonl \
      --batch-size 16 --mem-fraction 0.55
"""

import argparse
import json

from run_sweep import run_config

DEFAULT_TARGET = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_DRAFT = "rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="laptop_sweep.jsonl",
                        help="sweep output to find failed rows in")
    parser.add_argument("--output", default="bs16_retry.jsonl")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="only retry failed rows at this batch size")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--mem-fraction", type=float, default=0.55)
    parser.add_argument("--launch-timeout", type=int, default=900)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.source)]
    failed = [r for r in rows if r.get("batch_size") == args.batch_size and "error" in r]
    print(f"retrying {len(failed)} failed bs={args.batch_size} configs at mem-fraction={args.mem_fraction}")

    try:
        from components.profiler_cpu_gpu import HardwareMetricsProfiler
        profiler_instance = HardwareMetricsProfiler(gpu_index=0)
    except Exception as exc:
        print(f"profiler unavailable, continuing without energy metrics: {exc}")
        profiler_instance = None

    for i, r in enumerate(failed):
        cfg = (r["batch_size"], r["steps"], r["topk"], r["num_draft_tokens"])
        print(f"[{i+1}/{len(failed)}] orig index={r['index']} repeat={r['repeat']} "
              f"bs={cfg[0]} steps={cfg[1]} topk={cfg[2]} draft={cfg[3]} ...", flush=True)
        record = run_config(cfg, args, profiler_instance)
        record["index"] = r["index"]
        record["repeat"] = r["repeat"]
        record["retry_mem_fraction"] = args.mem_fraction
        with open(args.output, "a") as fout:
            fout.write(json.dumps(record) + "\n")
        if "error" in record:
            print(f"    ERROR {record['error']}", flush=True)
        else:
            print(f"    accept={record.get('accept_length')} "
                  f"speed={record.get('speed_tok_s')} tok/s "
                  f"J/tok={record.get('joules_per_token')}", flush=True)


if __name__ == "__main__":
    main()
