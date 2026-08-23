"""Run a small, telemetry-rich tau2-bench (Sierra Research) pilot.

This is a controlled subset evaluation, not an official tau2-bench score.
tau2's own `llm_agent` talks to a locally-launched SGLang server (the
repository's Llama + EAGLE3 pair) through its OpenAI-compatible endpoint,
using LiteLLM's ``api_base``/``api_key`` passthrough via ``--agent-llm-args``
-- no custom agent adapter is needed. tau2-bench itself must be installed
separately in an isolated venv (cloned from
``github.com/sierra-research/tau2-bench``, ``uv sync`` against Python
>=3.12); it is never a dependency of this repo.

Like the Terminal-Bench pilot (and unlike SWE-bench), tau2's `llm_agent`
owns the multi-turn user<->agent<->tools loop internally, so there is no
per-request telemetry hook. GPU/power/energy telemetry instead comes from a
continuous NVML sampler that runs for the duration of each ``tau2 run``
invocation; per-task numbers are derived post-hoc from each simulation's
``start_time``/``end_time`` plus the sampler's timestamped samples.

tau2-bench also requires a *second* LLM -- the user simulator that plays the
customer persona -- routed via OpenRouter (``OPENROUTER_API_KEY`` env var).
That key is never written to disk inside this repo; pass it in the
environment before running, e.g.:

  export OPENROUTER_API_KEY=...
  python RL/tau2bench_pilot.py prepare --root /tmp/tau2-bench-pilot-20260823/run
  python RL/tau2bench_pilot.py run --root /tmp/tau2-bench-pilot-20260823/run
  python RL/tau2bench_pilot.py report --root /tmp/tau2-bench-pilot-20260823/run

All large artifacts (results, telemetry, logs) should live under ``--root``,
outside the Git checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from components.profiler_cpu_gpu import HardwareMetricsProfiler
from run_sweep import (
    gpu_energy_mj,
    launch_server,
    read_server_info,
    wait_for_server,
)


DEFAULT_TARGET = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_DRAFT = "rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct"
DEFAULT_DOMAIN = "retail"
DEFAULT_TASK_IDS = ["105", "106"]
DEFAULT_USER_LLM = "openrouter/openai/gpt-4o-mini"
DEFAULT_ROOT = os.environ.get("T2B_PILOT_ROOT", "/tmp/tau2bench-pilot")
DEFAULT_TAU2_BIN = os.environ.get(
    "TAU2_BIN", "/tmp/tau2-bench-pilot-20260823/tau2-bench/.venv/bin/tau2"
)
DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
SERVER_BATCH_SIZE = 16
SCHEMA_VERSION = "tau2bench-pilot-v1"
SAMPLER_INTERVAL_S = 1.0

CONFIGS = [
    {"tag": "no_spec", "steps": 0, "topk": 0, "num_draft_tokens": 0},
    {"tag": "chosen", "steps": 3, "topk": 4, "num_draft_tokens": 8},
    {"tag": "chosen_bs16", "steps": 3, "topk": 2, "num_draft_tokens": 4},
]
CONFIG_BY_TAG = {config["tag"]: config for config in CONFIGS}


class PilotError(RuntimeError):
    """An expected pilot setup or execution error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PilotError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def sampler_path(root: Path, tag: str) -> Path:
    return root / "telemetry" / f"{tag}.samples.jsonl"


def config_summaries_path(root: Path) -> Path:
    return root / "telemetry" / "config_summaries.jsonl"


def save_to_path(root: Path, tag: str) -> Path:
    return root / "tau2-results" / tag


def results_file_path(root: Path, tag: str) -> Path:
    # tau2's CLI always nests the actual results at <save_to>/results.json
    # (run_domain -> _run_save_paths treats --save-to as run_name, joined
    # under DATA_DIR/simulations/ -- but joining an *absolute* save_to path
    # onto that base drops the prefix via pathlib's absolute-join behavior,
    # so results still land under our own root, just one level deeper).
    return save_to_path(root, tag) / "results.json"


def tau2_log_path(root: Path, tag: str) -> Path:
    return root / "logs" / f"{tag}.log"


def server_log_path(root: Path, tag: str) -> Path:
    return root / "logs" / f"{tag}.server.log"


def load_manifest(root: Path) -> dict[str, Any]:
    path = manifest_path(root)
    if not path.exists():
        raise PilotError(f"manifest is missing: run prepare first ({path})")
    with path.open() as handle:
        manifest = json.load(handle)
    if not manifest.get("task_ids"):
        raise PilotError(f"manifest has no task_ids: {path}")
    return manifest


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    if not Path(args.tau2_bin).exists():
        raise PilotError(f"tau2 binary not found: {args.tau2_bin}")
    help_result = subprocess.run(
        [args.tau2_bin, "run", "--help"], capture_output=True, text=True, timeout=30, check=False
    )
    checks["tau2_cli_ok"] = help_result.returncode == 0
    if not checks["tau2_cli_ok"]:
        raise PilotError(f"tau2 CLI is broken: {help_result.stderr.strip()}")

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise PilotError(
            "OPENROUTER_API_KEY is not set in the environment; the user-simulator "
            "LLM needs it. Export it before running prepare/run."
        )
    checks["openrouter_api_key_present"] = True

    checks["domain"] = args.domain
    checks["task_ids"] = args.task_ids

    return checks


def prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    checks = run_preflight(args)
    print("preflight checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created": utc_now(),
        "domain": args.domain,
        "task_ids": args.task_ids,
        "tau2_bin": args.tau2_bin,
        "target": args.target,
        "draft": args.draft,
        "user_llm": args.user_llm,
        "preflight": checks,
    }
    write_json(manifest_path(root), manifest)
    print(f"prepared manifest at {manifest_path(root)}")
    print(f"  domain={args.domain} task_ids={args.task_ids} user_llm={args.user_llm}")
    return manifest


def sampler_loop(profiler: Any, path: Path, stop_event: threading.Event, interval: float) -> None:
    while not stop_event.is_set():
        try:
            metrics = profiler.collect_metrics() if profiler else {}
            metrics = dict(metrics)
            metrics["gpu_total_energy_mj"] = gpu_energy_mj(profiler)
            append_jsonl(path, metrics)
        except Exception as exc:
            append_jsonl(path, {"timestamp": utc_now(), "sampler_error": f"{type(exc).__name__}: {exc}"})
        stop_event.wait(interval)


def start_sampler(profiler: Any, path: Path, interval: float = SAMPLER_INTERVAL_S) -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    thread = threading.Thread(target=sampler_loop, args=(profiler, path, stop_event, interval), daemon=True)
    thread.start()
    return thread, stop_event


def stop_sampler(thread: threading.Thread, stop_event: threading.Event, timeout: float = 10) -> None:
    stop_event.set()
    thread.join(timeout=timeout)


def stop_server(proc: Any) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def warmup(port: int, timeout: int) -> None:
    response = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "text": "Return the single word READY.",
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 8, "ignore_eos": False},
        },
        timeout=timeout,
    )
    response.raise_for_status()


def build_tau2_command(args: argparse.Namespace, config: dict[str, Any], root: Path) -> list[str]:
    model_string = f"openai/{args.target}"
    api_base = f"http://127.0.0.1:{args.port}/v1"
    agent_llm_args = json.dumps({"temperature": 0.0, "api_base": api_base, "api_key": "EMPTY"})
    command = [
        args.tau2_bin, "run",
        "--domain", args.domain,
        "--agent-llm", model_string,
        "--agent-llm-args", agent_llm_args,
        "--user-llm", args.user_llm,
        "--task-ids", *args.task_ids,
        "--num-trials", "1",
        "--max-steps", str(args.max_steps),
        "--max-errors", str(args.max_errors),
        "--timeout", str(args.sim_timeout),
        "--max-concurrency", str(args.max_concurrency),
        "--save-to", str(save_to_path(root, config["tag"])),
    ]
    return command


def run_config(args: argparse.Namespace, config: dict[str, Any], root: Path, profiler: Any) -> dict[str, Any]:
    tag = config["tag"]
    print(f"\n=== {tag} (steps={config['steps']} topk={config['topk']} draft={config['num_draft_tokens']}) ===", flush=True)
    proc = None
    server_started = time.perf_counter()
    sampler_thread = None
    sampler_stop = None
    error = None
    tau2_returncode = None
    accept_length = None
    try:
        server_log = server_log_path(root, tag)
        server_log.parent.mkdir(parents=True, exist_ok=True)
        proc = launch_server(
            args.target, args.draft, SERVER_BATCH_SIZE,
            config["steps"], config["topk"], config["num_draft_tokens"],
            args.port, args.mem_fraction, log_path=str(server_log),
            allow_auto_truncate=True,
        )
        if not wait_for_server(args.port, args.launch_timeout, proc):
            raise PilotError(f"server failed to start; see {server_log}")
        warmup(args.port, args.request_timeout)

        sampler_thread, sampler_stop = start_sampler(profiler, sampler_path(root, tag))

        command = build_tau2_command(args, config, root)
        env = dict(os.environ)
        # LiteLLM's OpenAI-compatible client for the local sglang server needs
        # this even though --agent-llm-args also carries api_key="EMPTY" --
        # same fix terminalbench_pilot.py needed for the Harbor/Terminus-2 case.
        env["OPENAI_API_KEY"] = "EMPTY"
        log_path = tau2_log_path(root, tag)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  running: {' '.join(command)}", flush=True)
        with log_path.open("w") as log_file:
            job = subprocess.run(
                command, env=env, stdout=log_file, stderr=subprocess.STDOUT,
                timeout=args.job_timeout, check=False,
            )
        tau2_returncode = job.returncode
        print(f"  tau2 exit={tau2_returncode}; log={log_path}", flush=True)

        try:
            accept_length, _ = read_server_info(args.port, SERVER_BATCH_SIZE)
        except Exception as exc:
            server_alive = proc.poll() is None
            print(
                f"  WARNING could not read /get_server_info (server_alive={server_alive}): "
                f"{type(exc).__name__}: {exc}; see {server_log}",
                flush=True,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"  ERROR {error}", flush=True)
    finally:
        if sampler_thread is not None:
            stop_sampler(sampler_thread, sampler_stop)
        stop_server(proc)
        time.sleep(5)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": utc_now(),
        "config": tag,
        "steps": config["steps"],
        "topk": config["topk"],
        "num_draft_tokens": config["num_draft_tokens"],
        "server_batch_size": SERVER_BATCH_SIZE,
        "tau2_returncode": tau2_returncode,
        "avg_spec_accept_length": round(accept_length, 5) if accept_length is not None else None,
        "wall_clock_s": round(time.perf_counter() - server_started, 4),
        "results_path": str(results_file_path(root, tag)),
        "sampler_path": str(sampler_path(root, tag)),
        "error": error,
    }
    return summary


def run_pilot(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    manifest = load_manifest(root)
    args.domain = manifest["domain"]
    if args.task_ids is None:
        args.task_ids = manifest["task_ids"]
    args.target = manifest["target"]
    args.draft = manifest["draft"]
    args.user_llm = manifest["user_llm"]

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise PilotError("OPENROUTER_API_KEY is not set in the environment")

    try:
        profiler = HardwareMetricsProfiler(gpu_index=0, output_dir=str(root / "profiler"))
    except Exception as exc:
        print(f"profiler unavailable; continuing without GPU energy metrics: {exc}")
        profiler = None

    configs = [CONFIG_BY_TAG[args.config]] if args.config else CONFIGS
    for config in configs:
        summary = run_config(args, config, root, profiler)
        append_jsonl(config_summaries_path(root), summary)


def load_results(path: Path) -> list[dict[str, Any]]:
    """SimulationRun entries from a tau2 `Results.simulations` list (results.json)."""
    if not path.exists():
        return []
    with path.open() as handle:
        value = json.load(handle)
    runs = []
    for run in value.get("simulations") or []:
        if isinstance(run, dict):
            run["_results_path"] = str(path)
            runs.append(run)
    return runs


def local_utc_offset() -> Any:
    return datetime.now().astimezone().utcoffset()


def parse_naive_local(timestamp: str, offset: Any) -> datetime:
    # tau2's get_now() writes naive datetime.now() timestamps (no tzinfo),
    # same as profiler_cpu_gpu.HardwareMetricsProfiler -- both are local time.
    return datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc) - offset


def segment_gpu_stats(sampler_path: Path, window_start: datetime, window_end: datetime) -> dict[str, Any] | None:
    if not sampler_path.exists() or window_end <= window_start:
        return None
    offset = local_utc_offset()
    samples = []
    for row in read_jsonl(sampler_path):
        ts = row.get("timestamp")
        if ts is None:
            continue
        sample_time = parse_naive_local(ts, offset)
        if window_start <= sample_time <= window_end:
            samples.append(row)
    if len(samples) < 2:
        return None
    powers = [s["gpu_power_watts"] for s in samples if s.get("gpu_power_watts") is not None]
    energies = [s["gpu_total_energy_mj"] for s in samples if s.get("gpu_total_energy_mj") is not None]
    temps = [s["gpu_temperature_celsius"] for s in samples if s.get("gpu_temperature_celsius") is not None]
    utils = [s["gpu_utilization_percent"] for s in samples if s.get("gpu_utilization_percent") is not None]
    return {
        "n_samples": len(samples),
        "avg_power_watts": round(float(np.mean(powers)), 3) if powers else None,
        "energy_delta_mj": round(energies[-1] - energies[0], 3) if len(energies) >= 2 else None,
        "avg_gpu_temperature_celsius": round(float(np.mean(temps)), 2) if temps else None,
        "avg_gpu_utilization_percent": round(float(np.mean(utils)), 2) if utils else None,
    }


def build_report_row(config_tag: str, config_summary: dict[str, Any] | None, run: dict[str, Any]) -> dict[str, Any]:
    start_time = run.get("start_time")
    end_time = run.get("end_time")
    duration_s = run.get("duration")
    tokens_per_second = None

    messages = run.get("messages") or []
    total_output_tokens = 0
    saw_usage = False
    for message in messages:
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is not None:
            saw_usage = True
            total_output_tokens += completion_tokens
    if not saw_usage:
        total_output_tokens = None
    if total_output_tokens is not None and duration_s:
        tokens_per_second = round(total_output_tokens / duration_s, 3)

    gpu_stats = None
    if config_summary and start_time and end_time:
        offset = local_utc_offset()
        gpu_stats = segment_gpu_stats(
            Path(config_summary["sampler_path"]),
            parse_naive_local(start_time, offset),
            parse_naive_local(end_time, offset),
        )

    reward_info = run.get("reward_info") or {}
    return {
        "config": config_tag,
        "task_id": run.get("task_id"),
        "trial": run.get("trial"),
        "reward": reward_info.get("reward"),
        "termination_reason": run.get("termination_reason"),
        "total_output_tokens": total_output_tokens,
        "duration_s": duration_s,
        "tokens_per_second": tokens_per_second,
        "avg_spec_accept_length": config_summary.get("avg_spec_accept_length") if config_summary else None,
        "gpu_stats": gpu_stats,
        "agent_cost": run.get("agent_cost"),
        "user_cost": run.get("user_cost"),
        "results_path": run["_results_path"],
    }


def report_results(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    manifest = load_manifest(root)
    config_summaries = {row["config"]: row for row in read_jsonl(config_summaries_path(root))}

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "domain": manifest["domain"],
        "task_ids": manifest["task_ids"],
        "configs": {},
    }
    merged_path = root / "reports" / "merged.jsonl"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w") as merged:
        for config in CONFIGS:
            tag = config["tag"]
            config_summary = config_summaries.get(tag)
            runs = load_results(results_file_path(root, tag))
            rows = [build_report_row(tag, config_summary, run) for run in runs]
            for row in rows:
                merged.write(json.dumps(row, default=str) + "\n")
            summary["configs"][tag] = {
                "config_summary": config_summary,
                "trial_count": len(rows),
                "rows": rows,
            }
    write_json(root / "reports" / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    print(f"merged report: {merged_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="run", choices=["prepare", "run", "report"])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--task-ids", nargs="+", default=None, help="defaults to manifest task_ids for run; DEFAULT_TASK_IDS for prepare")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--user-llm", default=DEFAULT_USER_LLM)
    parser.add_argument("--tau2-bin", default=DEFAULT_TAU2_BIN)
    parser.add_argument("--config", choices=[c["tag"] for c in CONFIGS], help="run only this config")
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--mem-fraction", type=float, default=0.55)
    parser.add_argument("--launch-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--job-timeout", type=int, default=1200)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-errors", type=int, default=5)
    parser.add_argument("--sim-timeout", type=int, default=300, help="tau2 --timeout: max wallclock seconds per simulation")
    parser.add_argument("--max-concurrency", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare" and args.task_ids is None:
        args.task_ids = DEFAULT_TASK_IDS
    if args.command == "prepare":
        prepare_manifest(args)
    elif args.command == "run":
        run_pilot(args)
    elif args.command == "report":
        report_results(args)


if __name__ == "__main__":
    main()
