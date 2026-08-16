"""Collect the offline dataset: one row per speculative decoding config.

Launches an SGLang server per config, benchmarks it, and records throughput
alongside energy/thermal metrics. The resulting JSONL is what the RL policy
trains against -- no server is ever launched from inside the training loop.

Measurement follows sglang's own scripts/playground/bench_speculative.py:
throughput is derived from /server_info rather than wall-clock timing, so the
numbers are directly comparable to sglang's published ones.

  speed = avg_spec_accept_length / p20(step_time)

Usage:
  export SGLANG_RECORD_STEP_TIME=1
  python3 run_sweep.py --output sweep.jsonl                # full 38-config grid
  python3 run_sweep.py --output sweep.jsonl --phase0       # baseline + reference only
  python3 run_sweep.py --output sweep.jsonl --start 25     # resume after preemption
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests

from sweep_config import BASELINE_CONFIG, REFERENCE_CONFIG, build_grid

DEFAULT_TARGET = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DEFAULT_DRAFT = "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"

# Matches bench_speculative.py: 512 output tokens over a fixed prompt set.
PROMPTS = [
    "Human: Give me a fully functional FastAPI server. Show the full, long python code without stop.\n\nAssistant:",
    "Human: Write a travel blog post to Hawaii.\n\nAssistant:",
    "Human: Solve x^2 = -1. Think step-by-step. Give me a long detailed explanation.\n\nAssistant:",
    "Human: Tell me about the president of the USA in wikipedia style.\n\nAssistant:",
]
OUTPUT_TOKENS = 512


def launch_server(target, draft, bs, steps, topk, num_draft, port, mem_fraction):
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", target,
        "--port", str(port),
        "--mem-fraction-static", str(mem_fraction),
        "--cuda-graph-max-bs", str(bs),
        "--max-running-requests", str(bs),
        "--log-level", "warning",
    ]
    if steps > 0:
        cmd += [
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", draft,
            "--speculative-num-steps", str(steps),
            "--speculative-eagle-topk", str(topk),
            "--speculative-num-draft-tokens", str(num_draft),
        ]
        if topk > 1:
            # server_args.py: topk > 1 with page_size > 1 needs flashinfer.
            cmd += ["--attention-backend", "flashinfer"]

    env = dict(os.environ)
    env["SGLANG_RECORD_STEP_TIME"] = "1"
    # The EAGLE3 draft head declares max_position_embeddings=2048.
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def wait_for_server(port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def send_batch(port, bs, num_prompts):
    """Fire num_prompts requests at concurrency bs. Returns elapsed seconds."""
    from concurrent.futures import ThreadPoolExecutor

    payloads = [
        {
            "text": PROMPTS[i % len(PROMPTS)],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": OUTPUT_TOKENS,
                "ignore_eos": True,
            },
        }
        for i in range(num_prompts)
    ]

    def one(p):
        return requests.post(f"http://127.0.0.1:{port}/generate", json=p, timeout=1800)

    start = time.time()
    with ThreadPoolExecutor(max_workers=bs) as pool:
        for r in pool.map(one, payloads):
            r.raise_for_status()
    return time.time() - start


def read_server_info(port, bs):
    info = requests.get(f"http://127.0.0.1:{port}/server_info", timeout=30).json()
    internal = info["internal_states"][0]
    accept_length = internal.get("avg_spec_accept_length") or 1.0
    step_times = internal.get("step_time_dict", {}).get(str(bs), [])
    return accept_length, step_times


def run_config(cfg, args, profiler_instance):
    bs, steps, topk, num_draft = cfg
    port = args.port

    proc = launch_server(args.target, args.draft, bs, steps, topk, num_draft,
                         port, args.mem_fraction)
    record = {
        "batch_size": bs, "steps": steps, "topk": topk,
        "num_draft_tokens": num_draft,
        "is_baseline": (steps, topk, num_draft) == BASELINE_CONFIG,
    }
    try:
        if not wait_for_server(port, args.launch_timeout):
            record["error"] = "server failed to start"
            return record

        send_batch(port, bs, bs)  # warmup

        metrics_before = profiler_instance.collect_metrics() if profiler_instance else {}
        energy_before = gpu_energy_mj(profiler_instance)

        elapsed = send_batch(port, bs, max(args.num_prompts, bs))

        energy_after = gpu_energy_mj(profiler_instance)
        metrics_after = profiler_instance.collect_metrics() if profiler_instance else {}

        accept_length, step_times = read_server_info(port, bs)
        import numpy as np
        # 20th percentile, as sglang does -- less sensitive to warmup outliers.
        step_time = float(np.percentile(step_times, 20)) if step_times else None

        total_tokens = max(args.num_prompts, bs) * OUTPUT_TOKENS
        record.update({
            "accept_length": round(accept_length, 3),
            "step_time_s": round(step_time, 6) if step_time else None,
            "speed_tok_s": round(accept_length / step_time, 3) if step_time else None,
            "wall_clock_s": round(elapsed, 3),
            "wall_clock_tok_s": round(total_tokens / elapsed, 3),
            "total_output_tokens": total_tokens,
        })

        if energy_before is not None and energy_after is not None:
            joules = (energy_after - energy_before) / 1000.0
            record["energy_joules"] = round(joules, 2)
            record["avg_power_watts"] = round(joules / elapsed, 2) if elapsed > 0 else None
            record["joules_per_token"] = round(joules / total_tokens, 5)

        if metrics_after:
            record["gpu_temp_c"] = metrics_after.get("gpu_temperature_celsius")
            record["gpu_power_w"] = metrics_after.get("gpu_power_watts")
            record["gpu_util_pct"] = metrics_after.get("gpu_utilization_percent")
            record["gpu_mem_used_mb"] = metrics_after.get("gpu_memory_used_mb")
            record["gpu_throttling"] = metrics_after.get("gpu_throttling_active")
            record["gpu_temp_c_before"] = metrics_before.get("gpu_temperature_celsius")
        return record
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        time.sleep(5)


def gpu_energy_mj(profiler_instance):
    if profiler_instance is None:
        return None
    try:
        import pynvml
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(profiler_instance.gpu_handle)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--output", default="sweep.jsonl")
    parser.add_argument("--start", type=int, default=0, help="resume index")
    parser.add_argument("--end", type=int)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--launch-timeout", type=int, default=900)
    parser.add_argument("--phase0", action="store_true",
                        help="only baseline + reference config at bs=1")
    args = parser.parse_args()

    if args.phase0:
        grid = [(1,) + BASELINE_CONFIG, (1,) + REFERENCE_CONFIG]
    else:
        grid = build_grid()

    end = args.end if args.end is not None else len(grid)
    todo = grid[args.start:end]

    try:
        from components.profiler_cpu_gpu import HardwareMetricsProfiler
        profiler_instance = HardwareMetricsProfiler(gpu_index=0)
    except Exception as exc:
        print(f"profiler unavailable, continuing without energy metrics: {exc}")
        profiler_instance = None

    print(f"running {len(todo)} configs (index {args.start}..{end - 1}) -> {args.output}")
    for offset, cfg in enumerate(todo):
        idx = args.start + offset
        print(f"[{idx}] bs={cfg[0]} steps={cfg[1]} topk={cfg[2]} draft={cfg[3]} ...", flush=True)
        record = run_config(cfg, args, profiler_instance)
        record["index"] = idx
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
