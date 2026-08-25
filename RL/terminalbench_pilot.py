"""Run a small, telemetry-rich Terminal-Bench pilot via Harbor + Terminus 2.

This is a controlled subset evaluation, not an official Terminal-Bench score.
Terminus 2 (Harbor's reference agent) talks to a locally-launched SGLang
server (the repository's Llama + EAGLE3 pair) through its OpenAI-compatible
endpoint, using LiteLLM's ``api_base`` support -- no custom agent adapter is
needed. Harbor itself must be installed separately in an isolated venv
(``harbor`` on PyPI, Python >=3.12); it is never a dependency of this repo.

Unlike the SWE-bench pilot, Harbor/Terminus 2 owns the multi-turn tool loop
internally, so there is no per-request telemetry hook. GPU/power/energy
telemetry instead comes from a continuous NVML sampler that runs for the
duration of each ``harbor jobs start`` invocation; per-task numbers are
derived post-hoc from Harbor's own trial timing plus the sampler's
timestamped samples.

All large artifacts (Harbor venv, job outputs, telemetry, logs) should live
under ``--root``, outside the Git checkout.

Examples:
  python RL/terminalbench_pilot.py prepare --root /tmp/terminalbench-pilot
  python RL/terminalbench_pilot.py run --root /tmp/terminalbench-pilot
  python RL/terminalbench_pilot.py report --root /tmp/terminalbench-pilot
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
DEFAULT_DATASET = "terminal-bench-sample@2.0"
DEFAULT_TASKS = ["regex-log", "log-summary-date-ranges"]
# Real limits for the local sglang server, so Terminus 2 compacts instead of
# overflowing. max_input_tokens matches the server's max_req_input_len.
MODEL_INFO = {
    "max_input_tokens": 57760,
    "max_output_tokens": 4096,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
DEFAULT_ROOT = os.environ.get("TB_PILOT_ROOT", "/tmp/terminalbench-pilot")
DEFAULT_HARBOR_BIN = os.environ.get(
    "HARBOR_BIN", "/tmp/terminal-bench-eval-20260823/bin/harbor"
)
DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
SERVER_BATCH_SIZE = 16
SCHEMA_VERSION = "terminalbench-pilot-v1"
SAMPLER_INTERVAL_S = 1.0
MIN_FREE_DISK_GB = 3.0

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


def harbor_jobs_dir(root: Path, tag: str) -> Path:
    return root / "harbor-jobs" / tag


def harbor_log_path(root: Path, tag: str) -> Path:
    return root / "logs" / f"{tag}.log"


def server_log_path(root: Path, tag: str) -> Path:
    return root / "logs" / f"{tag}.server.log"


def load_manifest(root: Path) -> dict[str, Any]:
    path = manifest_path(root)
    if not path.exists():
        raise PilotError(f"manifest is missing: run prepare first ({path})")
    with path.open() as handle:
        manifest = json.load(handle)
    if not manifest.get("tasks"):
        raise PilotError(f"manifest has no tasks: {path}")
    return manifest


def harbor_python(harbor_bin: str) -> Path:
    return Path(harbor_bin).resolve().parent / "python"


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    docker_result = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=15, check=False
    )
    checks["docker_daemon_reachable"] = docker_result.returncode == 0
    if not checks["docker_daemon_reachable"]:
        raise PilotError(f"Docker daemon is not reachable: {docker_result.stderr.strip()}")

    usage = shutil.disk_usage(Path(args.root).expanduser().resolve().anchor or "/")
    free_gb = usage.free / (1024**3)
    checks["free_disk_gb"] = round(free_gb, 2)
    if free_gb < MIN_FREE_DISK_GB:
        print(f"WARNING: only {free_gb:.2f} GB free disk; Harbor task images may not fit")

    if not Path(args.harbor_bin).exists():
        raise PilotError(f"harbor binary not found: {args.harbor_bin}")
    help_result = subprocess.run(
        [args.harbor_bin, "jobs", "start", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    checks["harbor_cli_ok"] = help_result.returncode == 0
    if not checks["harbor_cli_ok"]:
        raise PilotError(f"harbor CLI is broken: {help_result.stderr.strip()}")

    python_path = harbor_python(args.harbor_bin)
    version_result = subprocess.run(
        [str(python_path), "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    version_str = version_result.stdout.strip()
    checks["harbor_python_version"] = version_str
    try:
        major, minor = (int(part) for part in version_str.split("."))
    except ValueError:
        major, minor = 0, 0
    if (major, minor) < (3, 12):
        raise PilotError(f"harbor venv Python must be >=3.12, found {version_str!r}")

    version_out = subprocess.run(
        [args.harbor_bin, "--version"], capture_output=True, text=True, timeout=15, check=False
    )
    checks["harbor_version"] = version_out.stdout.strip()

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
        "dataset": args.dataset,
        "tasks": args.tasks or DEFAULT_TASKS,
        "harbor_bin": args.harbor_bin,
        "target": args.target,
        "draft": args.draft,
        "preflight": checks,
    }
    write_json(manifest_path(root), manifest)
    print(f"prepared manifest at {manifest_path(root)}")
    print(f"  dataset={args.dataset} tasks={args.tasks}")
    return manifest


def sampler_loop(
    profiler: Any,
    path: Path,
    stop_event: threading.Event,
    interval: float,
) -> None:
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
    thread = threading.Thread(
        target=sampler_loop, args=(profiler, path, stop_event, interval), daemon=True
    )
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


def build_harbor_command(args: argparse.Namespace, config: dict[str, Any], root: Path) -> list[str]:
    model_string = f"openai/{args.target}"
    api_base = f"http://127.0.0.1:{args.port}/v1"
    command = [
        args.harbor_bin, "jobs", "start",
        "-d", args.dataset,
        "-a", "terminus-2",
        "-m", model_string,
        "--ak", f"api_base={api_base}",
        "--ak", "temperature=0",
        # Without this, LiteLLM can't map the model and Terminus 2 falls back to a
        # 1,000,000-token context limit, so it never compacts. The prompt then grows
        # past the server's real max_req_input_len (57,760) and every request gets
        # truncated server-side until the task stalls out.
        "--ak", f"model_info={json.dumps(MODEL_INFO)}",
        "-n", "1",
        "-o", str(harbor_jobs_dir(root, config["tag"])),
        "--job-name", config["tag"],
        "--yes",
    ]
    for task in args.tasks:
        command += ["-i", task]
    return command


def run_config(args: argparse.Namespace, config: dict[str, Any], root: Path, profiler: Any) -> dict[str, Any]:
    tag = config["tag"]
    print(f"\n=== {tag} (steps={config['steps']} topk={config['topk']} draft={config['num_draft_tokens']}) ===", flush=True)
    proc = None
    server_started = time.perf_counter()
    sampler_thread = None
    sampler_stop = None
    error = None
    harbor_returncode = None
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

        command = build_harbor_command(args, config, root)
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = "EMPTY"
        log_path = harbor_log_path(root, tag)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  running: {' '.join(command)}", flush=True)
        with log_path.open("w") as log_file:
            job = subprocess.run(
                command, env=env, stdout=log_file, stderr=subprocess.STDOUT,
                timeout=args.job_timeout, check=False,
            )
        harbor_returncode = job.returncode
        print(f"  harbor exit={harbor_returncode}; log={log_path}", flush=True)

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
        "harbor_returncode": harbor_returncode,
        "avg_spec_accept_length": round(accept_length, 5) if accept_length is not None else None,
        "wall_clock_s": round(time.perf_counter() - server_started, 4),
        "jobs_dir": str(harbor_jobs_dir(root, tag)),
        "sampler_path": str(sampler_path(root, tag)),
        "error": error,
    }
    return summary


def run_pilot(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    manifest = load_manifest(root)
    args.dataset = manifest["dataset"]
    if args.tasks is None:
        args.tasks = manifest["tasks"]
    args.target = manifest["target"]
    args.draft = manifest["draft"]

    try:
        profiler = HardwareMetricsProfiler(gpu_index=0, output_dir=str(root / "profiler"))
    except Exception as exc:
        print(f"profiler unavailable; continuing without GPU energy metrics: {exc}")
        profiler = None

    configs = [CONFIG_BY_TAG[args.config]] if args.config else CONFIGS
    for config in configs:
        summary = run_config(args, config, root, profiler)
        append_jsonl(config_summaries_path(root), summary)


def find_trial_results(jobs_dir: Path) -> list[dict[str, Any]]:
    """Trial-level result.json files only (job-level ones lack "verifier_result")."""
    results = []
    if not jobs_dir.exists():
        return results
    for path in jobs_dir.rglob("result.json"):
        try:
            with path.open() as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if "verifier_result" not in value:
            continue
        value["_result_path"] = str(path)
        results.append(value)
    return results


def local_utc_offset() -> Any:
    return datetime.now().astimezone().utcoffset()


def parse_utc(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def parse_sampler_timestamp(timestamp: str, offset: Any) -> datetime:
    # profiler_cpu_gpu.HardwareMetricsProfiler writes naive datetime.now() timestamps.
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
        sample_time = parse_sampler_timestamp(ts, offset)
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


def build_report_row(config_tag: str, config_summary: dict[str, Any] | None, trial: dict[str, Any], root: Path) -> dict[str, Any]:
    agent_result = trial.get("agent_result") or {}
    verifier_result = trial.get("verifier_result") or {}
    exception_info = trial.get("exception_info")
    agent_execution = trial.get("agent_execution") or {}

    n_output_tokens = agent_result.get("n_output_tokens")
    duration_s = None
    tokens_per_second = None
    if agent_execution.get("started_at") and agent_execution.get("finished_at"):
        start = parse_utc(agent_execution["started_at"])
        end = parse_utc(agent_execution["finished_at"])
        duration_s = (end - start).total_seconds()
        if n_output_tokens is not None and duration_s > 0:
            tokens_per_second = round(n_output_tokens / duration_s, 3)

    gpu_stats = None
    if config_summary and duration_s is not None:
        gpu_stats = segment_gpu_stats(
            Path(config_summary["sampler_path"]),
            parse_utc(agent_execution["started_at"]),
            parse_utc(agent_execution["finished_at"]),
        )

    rewards = verifier_result.get("rewards") or {}
    return {
        "config": config_tag,
        "task_name": trial.get("task_name"),
        "trial_name": trial.get("trial_name"),
        "reward": rewards.get("reward"),
        "n_input_tokens": agent_result.get("n_input_tokens"),
        "n_output_tokens": n_output_tokens,
        "n_episodes": (agent_result.get("metadata") or {}).get("n_episodes"),
        "agent_execution_duration_s": round(duration_s, 3) if duration_s is not None else None,
        "tokens_per_second": tokens_per_second,
        "avg_spec_accept_length": config_summary.get("avg_spec_accept_length") if config_summary else None,
        "gpu_stats": gpu_stats,
        "exception_type": (exception_info or {}).get("exception_type"),
        "exception_message": (exception_info or {}).get("exception_message"),
        "result_path": trial["_result_path"],
    }


def report_results(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    manifest = load_manifest(root)
    config_summaries = {row["config"]: row for row in read_jsonl(config_summaries_path(root))}

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "dataset": manifest["dataset"],
        "tasks": manifest["tasks"],
        "configs": {},
    }
    merged_path = root / "reports" / "merged.jsonl"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w") as merged:
        for config in CONFIGS:
            tag = config["tag"]
            config_summary = config_summaries.get(tag)
            trials = find_trial_results(harbor_jobs_dir(root, tag))
            rows = [build_report_row(tag, config_summary, trial, root) for trial in trials]
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

    pulled = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    tb_images = [line for line in pulled.stdout.splitlines() if "laude-institute/terminal-bench" in line]
    if tb_images:
        print("\nDocker images pulled for this pilot (confirm exact names before deleting):")
        for image in tb_images:
            print(f"  {image}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="run", choices=["prepare", "run", "report"])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--tasks", nargs="+", default=None, help="defaults to manifest tasks for run; DEFAULT_TASKS for prepare")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--harbor-bin", default=DEFAULT_HARBOR_BIN)
    parser.add_argument("--config", choices=[c["tag"] for c in CONFIGS], help="run only this config")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction", type=float, default=0.55)
    parser.add_argument("--launch-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--job-timeout", type=int, default=3600)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_manifest(args)
    elif args.command == "run":
        run_pilot(args)
    elif args.command == "report":
        report_results(args)


if __name__ == "__main__":
    main()
