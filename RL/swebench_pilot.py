"""Run a small, telemetry-rich SWE-bench Lite pilot.

This is a controlled subset evaluation, not a full SWE-bench score.  It uses
one deterministic patch-generation request per instance/configuration, the
repository's Llama + EAGLE3 pair, and the official SWE-bench Docker harness.
All large artifacts should live under ``--root`` outside the Git checkout.

Examples:
  python RL/swebench_pilot.py prepare --root /tmp/swebench-lite-pilot
  python RL/swebench_pilot.py generate --root /tmp/swebench-lite-pilot
  python RL/swebench_pilot.py validate --root /tmp/swebench-lite-pilot
  python RL/swebench_pilot.py evaluate --root /tmp/swebench-lite-pilot \
      --evaluator-python /tmp/swebench-eval/bin/python
  python RL/swebench_pilot.py report --root /tmp/swebench-lite-pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
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
DEFAULT_DATASET = "SWE-bench/SWE-bench_Lite"
DEFAULT_ROOT = os.environ.get("SWE_PILOT_ROOT", "/tmp/swebench-lite-pilot")
DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
SERVER_BATCH_SIZE = 16
EFFECTIVE_CONCURRENCY = 1
SCHEMA_VERSION = "swebench-pilot-v1"

CONFIGS = [
    {
        "tag": "no_spec",
        "steps": 0,
        "topk": 0,
        "num_draft_tokens": 0,
        "model_name": "energy-llms-swebench-no-spec",
    },
    {
        "tag": "chosen",
        "steps": 3,
        "topk": 4,
        "num_draft_tokens": 8,
        "model_name": "energy-llms-swebench-chosen-3-4-8",
    },
    {
        "tag": "chosen_bs16",
        "steps": 3,
        "topk": 2,
        "num_draft_tokens": 4,
        "model_name": "energy-llms-swebench-chosen-bs16-3-2-4",
    },
]
CONFIG_BY_TAG = {config["tag"]: config for config in CONFIGS}


class PilotError(RuntimeError):
    """An expected pilot setup or execution error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(
    command: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=capture,
        check=check,
        timeout=timeout,
    )


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


def latest_by_key(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        if all(value is not None for value in key):
            latest[key] = row
    return latest


def manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def telemetry_path(root: Path, tag: str) -> Path:
    return root / "telemetry" / f"{tag}.jsonl"


def prediction_path(root: Path, tag: str) -> Path:
    return root / "predictions" / f"{tag}.jsonl"


def checkout_path(root: Path, repo: str) -> Path:
    return root / "repositories" / repo.replace("/", "__")


def load_manifest(root: Path) -> list[dict[str, Any]]:
    path = manifest_path(root)
    if not path.exists():
        raise PilotError(f"manifest is missing: run prepare first ({path})")
    with path.open() as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, list) or not manifest:
        raise PilotError(f"manifest must be a non-empty list: {path}")
    return manifest


def select_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    requested_ids: list[str],
    requested_repo: str | None,
) -> list[dict[str, Any]]:
    by_id = {row["instance_id"]: row for row in rows}
    if requested_ids:
        missing = [instance_id for instance_id in requested_ids if instance_id not in by_id]
        if missing:
            raise PilotError(f"instance IDs are not in the dataset: {missing}")
        selected = [by_id[instance_id] for instance_id in requested_ids]
    else:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if requested_repo is None or row.get("repo") == requested_repo:
                groups[row["repo"]].append(row)
        # Prefer one repository so the Docker/source checkout cost is shared.
        eligible = [group for group in groups.values() if len(group) >= limit]
        if eligible:
            selected = eligible[0][:limit]
        else:
            selected = [row for group in groups.values() for row in group][:limit]
    if not selected:
        raise PilotError("the dataset selection is empty")
    return selected


def prepare_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    rows = [dict(row) for row in dataset]
    selected = select_dataset_rows(
        rows,
        limit=args.limit,
        requested_ids=args.instance_id,
        requested_repo=args.repo,
    )
    manifest = []
    for row in selected:
        manifest.append(
            {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "problem_statement": row["problem_statement"],
                "hints_text": row.get("hints_text") or "",
            }
        )
    write_json(manifest_path(root), manifest)
    for item in manifest:
        ensure_checkout(root, item["repo"], item["base_commit"])
    print(f"prepared {len(manifest)} instances in {manifest_path(root)}")
    for item in manifest:
        print(f"  {item['instance_id']}  ({item['repo']} @ {item['base_commit']})")
    return manifest


def ensure_checkout(root: Path, repo: str, base_commit: str) -> Path:
    path = checkout_path(root, repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        if path.exists() and any(path.iterdir()):
            raise PilotError(f"checkout path is not an empty Git checkout: {path}")
        run_cmd(["git", "clone", "--filter=blob:none", f"https://github.com/{repo}.git", str(path)], timeout=1800)
    run_cmd(["git", "fetch", "--depth", "1", "origin", base_commit], cwd=path, timeout=1800)
    run_cmd(["git", "checkout", "--detach", base_commit], cwd=path, timeout=300)
    return path


def issue_keywords(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text.lower())
    stop = {
        "this", "that", "with", "from", "have", "when", "which", "should",
        "does", "into", "there", "their", "will", "would", "could", "being",
        "expected", "actual", "problem", "issue", "please", "using", "error",
    }
    return {word for word in words if word not in stop}


def build_source_context(checkout: Path, issue: str, max_chars: int = 24000) -> str:
    tree = run_cmd(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=checkout).stdout.splitlines()
    keywords = issue_keywords(issue)

    def score(path: str) -> tuple[int, int, str]:
        lower = path.lower()
        hits = sum(keyword in lower for keyword in keywords)
        preferred = int(any(name in lower for name in ("readme", "setup.py", "pyproject.toml", "__init__.py")))
        return (-hits, -preferred, path)

    matching = [path for path in tree if any(keyword in path.lower() for keyword in keywords)]
    candidates = []
    seen = set()
    for path in matching + sorted(tree, key=score):
        if path in seen or path.endswith((".pyc", ".so", ".png", ".jpg", ".gif", ".lock")):
            continue
        seen.add(path)
        candidates.append(path)
        if len(candidates) >= 24:
            break

    sections = ["Repository file tree:\n" + "\n".join(tree[:500])]
    used = len(sections[0])
    for path in candidates:
        if used >= max_chars:
            break
        result = run_cmd(["git", "show", f"HEAD:{path}"], cwd=checkout, check=False)
        if result.returncode != 0 or "\x00" in result.stdout:
            continue
        content = result.stdout[:5000]
        section = f"\n\n--- {path} ---\n{content}"
        if used + len(section) > max_chars:
            section = section[: max_chars - used]
        sections.append(section)
        used += len(section)
    return "".join(sections)


def format_chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    if tokenizer is None:
        return user_prompt + "\n\nAssistant:"
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return user_prompt + "\n\nAssistant:"


def load_tokenizer(target: str) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(target, trust_remote_code=True)
    except Exception as exc:
        print(f"tokenizer unavailable; using plain prompt and metadata counts: {exc}")
        return None


def make_prompt(item: dict[str, Any], checkout: Path, tokenizer: Any) -> str:
    issue = item["problem_statement"].strip()
    hints = item.get("hints_text", "").strip()
    context = build_source_context(checkout, issue)
    prompt = f"""You are repairing a real software issue at the repository's exact base commit.

Instance: {item['instance_id']}
Repository: {item['repo']}
Base commit: {item['base_commit']}

Issue report:
{issue}
"""
    if hints:
        prompt += f"\nAdditional maintainer hints:\n{hints}\n"
    prompt += f"""
Relevant repository context:
{context}

Produce the smallest fix that resolves the issue. Return ONLY a unified git diff
that applies cleanly at the specified base commit. Do not return Markdown fences,
explanations, test output, or commentary. If no change is needed, still return a
valid empty-context diff rather than prose.
"""
    return format_chat_prompt(tokenizer, prompt)


def request_generation(port: int, prompt: str, max_new_tokens: int, timeout: int) -> tuple[str, dict[str, Any]]:
    response = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": False,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload.get("text") or payload.get("output") or ""
    meta = payload.get("meta_info") or {}
    if not isinstance(meta, dict):
        meta = {}
    return text, meta


def token_count(tokenizer: Any, text: str) -> int | None:
    if tokenizer is None:
        return None
    try:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:
        return None


def extract_patch(text: str) -> str | None:
    cleaned = text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
    fenced = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    starts = [match.start() for match in re.finditer(r"(?m)^diff --git ", cleaned)]
    if starts:
        cleaned = cleaned[min(starts):]
    else:
        header = re.search(r"(?m)^--- (?:a/|/dev/null)", cleaned)
        if not header:
            return None
        cleaned = cleaned[header.start():]
    cleaned = cleaned.split("\n```", 1)[0].strip()
    return cleaned or None


def patch_applies(checkout: Path, patch: str) -> tuple[bool, str]:
    result = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=checkout,
        input=patch,
        text=True,
        capture_output=True,
        check=False,
    )
    detail = (result.stderr or result.stdout or "").strip()
    return result.returncode == 0, detail


def fallback_patch(instance_id: str, config_tag: str) -> str:
    digest = hashlib.sha256(f"{instance_id}:{config_tag}".encode()).hexdigest()[:12]
    path = f".swebench_pilot_noop_{digest}.txt"
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1 @@\n"
        "+SWE-bench pilot fallback\n"
    )


def valid_or_fallback(
    checkout: Path,
    candidate: str | None,
    instance_id: str,
    config_tag: str,
) -> tuple[str, str, str | None]:
    if candidate:
        applies, detail = patch_applies(checkout, candidate)
        if applies:
            return candidate, "model_valid", None
    else:
        detail = "model output did not contain a unified diff"
    fallback = fallback_patch(instance_id, config_tag)
    applies, fallback_detail = patch_applies(checkout, fallback)
    if not applies:
        raise PilotError(f"internal fallback patch does not apply: {fallback_detail}")
    return fallback, "model_invalid_fallback", detail


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
    request_generation(
        port,
        "Return the single word READY.",
        max_new_tokens=8,
        timeout=timeout,
    )


def metrics_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    names = (
        "gpu_temperature_celsius", "gpu_power_watts", "gpu_utilization_percent",
        "gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_throttling_active",
        "gpu_sm_clock_mhz", "gpu_memory_clock_mhz", "cpu_utilization_percent",
        "system_memory_used_mb", "timestamp",
    )
    return {name: metrics.get(name) for name in names if name in metrics}


def generate_one(
    *,
    args: argparse.Namespace,
    item: dict[str, Any],
    config: dict[str, Any],
    checkout: Path,
    tokenizer: Any,
    profiler: Any,
    port: int,
) -> dict[str, Any]:
    prompt = make_prompt(item, checkout, tokenizer)
    metrics_before = profiler.collect_metrics() if profiler else {}
    energy_before = gpu_energy_mj(profiler)
    started = time.perf_counter()
    output = ""
    meta: dict[str, Any] = {}
    request_error = None
    try:
        output, meta = request_generation(port, prompt, args.max_new_tokens, args.request_timeout)
    except Exception as exc:
        request_error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    energy_after = gpu_energy_mj(profiler)
    metrics_after = profiler.collect_metrics() if profiler else {}

    prompt_tokens = meta.get("prompt_tokens")
    completion_tokens = meta.get("completion_tokens")
    token_source = "sglang_meta_info"
    if not isinstance(prompt_tokens, int):
        prompt_tokens = token_count(tokenizer, prompt)
        token_source = "tokenizer_fallback"
    if not isinstance(completion_tokens, int):
        completion_tokens = token_count(tokenizer, output)
        token_source = "tokenizer_fallback"
    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    patch = None
    patch_status = "request_failed_fallback"
    patch_error = request_error
    if request_error is None:
        patch, patch_status, patch_error = valid_or_fallback(
            checkout, extract_patch(output), item["instance_id"], config["tag"]
        )
    else:
        patch, _, _ = valid_or_fallback(
            checkout, None, item["instance_id"], config["tag"]
        )
        patch_status = "request_failed_fallback"

    energy_joules = None
    energy_source = "unavailable"
    if energy_before is not None and energy_after is not None and energy_after >= energy_before:
        energy_joules = (energy_after - energy_before) / 1000.0
        energy_source = "nvml_total_energy"

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "timestamp": utc_now(),
        "config": config["tag"],
        "steps": config["steps"],
        "topk": config["topk"],
        "num_draft_tokens": config["num_draft_tokens"],
        "instance_id": item["instance_id"],
        "repo": item["repo"],
        "base_commit": item["base_commit"],
        "target_model": args.target,
        "draft_model": args.draft if config["steps"] else None,
        "server_batch_size": SERVER_BATCH_SIZE,
        "effective_concurrency": EFFECTIVE_CONCURRENCY,
        "temperature": 0.0,
        "top_p": 1.0,
        "status": "completed" if request_error is None else "request_error",
        "error_type": type(request_error).__name__ if request_error else None,
        "error": request_error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or None,
        "token_count_source": token_source,
        "model_elapsed_s": round(elapsed, 6),
        "model_elapsed_ms": round(elapsed * 1000, 3),
        "output_tokens_per_s": round(completion_tokens / elapsed, 3) if completion_tokens and elapsed > 0 else None,
        "energy_before_mj": energy_before,
        "energy_after_mj": energy_after,
        "energy_joules": round(energy_joules, 4) if energy_joules is not None else None,
        "avg_power_watts": round(energy_joules / elapsed, 3) if energy_joules is not None and elapsed > 0 else None,
        "joules_per_output_token": round(energy_joules / completion_tokens, 6) if energy_joules is not None and completion_tokens else None,
        "energy_source": energy_source,
        "gpu_metrics_before": metrics_subset(metrics_before),
        "gpu_metrics_after": metrics_subset(metrics_after),
        "patch_generated": bool(output),
        "patch_applied": patch_status == "model_valid",
        "patch_status": patch_status,
        "patch_error": patch_error,
        "tests_ran": False,
        "resolved": None,
        "fail_to_pass": None,
        "pass_to_pass": None,
        "model_patch": patch,
    }
    return record


def config_summary(
    config: dict[str, Any],
    *,
    port: int,
    profiler: Any,
    started: float,
    error: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": utc_now(),
        "config": config["tag"],
        "steps": config["steps"],
        "topk": config["topk"],
        "num_draft_tokens": config["num_draft_tokens"],
        "server_batch_size": SERVER_BATCH_SIZE,
        "effective_concurrency": EFFECTIVE_CONCURRENCY,
        "error": error,
    }
    if error is not None:
        return summary
    try:
        accept_length, step_times = read_server_info(port, SERVER_BATCH_SIZE)
        step_time = float(np.percentile(step_times, 20)) if step_times else None
        summary.update(
            {
                "avg_spec_accept_length": round(accept_length, 5),
                "step_time_p20_s": round(step_time, 6) if step_time is not None else None,
                "server_speed_tok_s": round(accept_length / step_time, 4) if step_time else None,
            }
        )
    except Exception as exc:
        summary["error"] = f"server info: {type(exc).__name__}: {exc}"
    summary["wall_clock_s"] = round(time.perf_counter() - started, 4)
    return summary


def write_predictions(root: Path, config: dict[str, Any], manifest: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows = latest_by_key(read_jsonl(telemetry_path(root, config["tag"])), ("instance_id",))
    path = prediction_path(root, config["tag"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for item in manifest:
            row = rows.get((item["instance_id"],))
            if not row:
                continue
            handle.write(
                json.dumps(
                    {
                        "instance_id": item["instance_id"],
                        "model_name_or_path": config["model_name"],
                        "model_patch": row["model_patch"],
                    }
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def generate_predictions(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    manifest = load_manifest(root)
    tokenizer = load_tokenizer(args.target)
    try:
        profiler = HardwareMetricsProfiler(gpu_index=0, output_dir=str(root / "profiler"))
    except Exception as exc:
        print(f"profiler unavailable; continuing without GPU energy metrics: {exc}")
        profiler = None

    for config in CONFIGS:
        path = telemetry_path(root, config["tag"])
        latest = latest_by_key(read_jsonl(path), ("instance_id",))
        pending = []
        for item in manifest:
            previous = latest.get((item["instance_id"],))
            if previous and previous.get("status") == "completed" and not args.retry_errors:
                continue
            if previous and previous.get("status") != "completed" and not args.retry_errors:
                continue
            pending.append(item)
        if not pending:
            write_predictions(root, config, manifest, args)
            print(f"{config['tag']}: already complete")
            continue

        print(f"\n=== {config['tag']} ({len(pending)} instances) ===", flush=True)
        proc = None
        server_started = time.perf_counter()
        try:
            proc = launch_server(
                args.target,
                args.draft,
                SERVER_BATCH_SIZE,
                config["steps"],
                config["topk"],
                config["num_draft_tokens"],
                args.port,
                args.mem_fraction,
            )
            if not wait_for_server(args.port, args.launch_timeout, proc):
                raise PilotError("server failed to start")
            warmup(args.port, args.request_timeout)
            for item in pending:
                print(f"  {item['instance_id']} ...", flush=True)
                record = generate_one(
                    args=args,
                    item=item,
                    config=config,
                    checkout=checkout_path(root, item["repo"]),
                    tokenizer=tokenizer,
                    profiler=profiler,
                    port=args.port,
                )
                append_jsonl(path, record)
                print(
                    f"    status={record['status']} patch={record['patch_status']} "
                    f"tokens={record['completion_tokens']} tps={record['output_tokens_per_s']} "
                    f"J={record['energy_joules']}",
                    flush=True,
                )
            summary = config_summary(config, port=args.port, profiler=profiler, started=server_started)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR {detail}", flush=True)
            for item in pending:
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": args.run_id,
                    "timestamp": utc_now(),
                    "config": config["tag"],
                    "steps": config["steps"],
                    "topk": config["topk"],
                    "num_draft_tokens": config["num_draft_tokens"],
                    "instance_id": item["instance_id"],
                    "repo": item["repo"],
                    "base_commit": item["base_commit"],
                    "target_model": args.target,
                    "draft_model": args.draft if config["steps"] else None,
                    "server_batch_size": SERVER_BATCH_SIZE,
                    "effective_concurrency": EFFECTIVE_CONCURRENCY,
                    "status": "server_start_error",
                    "error_type": type(exc).__name__,
                    "error": detail,
                    "patch_generated": False,
                    "patch_applied": False,
                    "patch_status": "request_failed_fallback",
                    "patch_error": detail,
                    "tests_ran": False,
                    "resolved": None,
                    "model_patch": fallback_patch(item["instance_id"], config["tag"]),
                }
                append_jsonl(path, record)
            summary = config_summary(config, port=args.port, profiler=profiler, started=server_started, error=detail)
        finally:
            stop_server(proc)
            time.sleep(5)
        append_jsonl(root / "telemetry" / "config_summaries.jsonl", summary)
        write_predictions(root, config, manifest, args)


def validate_predictions(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    manifest = load_manifest(root)
    by_id = {item["instance_id"]: item for item in manifest}
    for config in CONFIGS:
        path = prediction_path(root, config["tag"])
        if not path.exists():
            raise PilotError(f"missing prediction file: {path}")
        telemetry = latest_by_key(read_jsonl(telemetry_path(root, config["tag"])), ("instance_id",))
        rows = read_jsonl(path)
        seen = set()
        for row in rows:
            required = {"instance_id", "model_name_or_path", "model_patch"}
            missing = required - row.keys()
            if missing:
                raise PilotError(f"{path} row missing {sorted(missing)}")
            instance_id = row["instance_id"]
            if instance_id not in by_id:
                raise PilotError(f"unexpected instance ID in {path}: {instance_id}")
            if instance_id in seen:
                raise PilotError(f"duplicate instance ID in {path}: {instance_id}")
            patch = row["model_patch"]
            if not isinstance(patch, str) or not patch.strip():
                raise PilotError(f"empty model patch in {path}: {instance_id}")
            applies, detail = patch_applies(checkout_path(root, by_id[instance_id]["repo"]), patch)
            if not applies:
                raise PilotError(f"patch does not apply for {instance_id} in {path}: {detail}")
            if (instance_id,) not in telemetry:
                raise PilotError(f"prediction has no telemetry row for {instance_id}: {path}")
            seen.add(instance_id)
        missing_ids = set(by_id) - seen
        if missing_ids:
            raise PilotError(f"{path} is missing instances: {sorted(missing_ids)}")
        print(f"validated {path} ({len(rows)} predictions; patches apply)")


def evaluator_help(eval_python: str) -> str:
    result = run_cmd(
        [eval_python, "-m", "swebench.harness.run_evaluation", "--help"],
        check=False,
        timeout=120,
    )
    return (result.stdout or "") + (result.stderr or "")


def run_evaluator_command(
    args: argparse.Namespace,
    command: list[str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    print(f"evaluator exit={process.returncode}; log={log_path}")
    return process.returncode


def evaluate_predictions(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    manifest = load_manifest(root)
    eval_python = args.evaluator_python or os.environ.get("SWE_EVAL_PYTHON")
    if not eval_python:
        raise PilotError("pass --evaluator-python or set SWE_EVAL_PYTHON")
    if not Path(eval_python).exists():
        raise PilotError(f"evaluator Python does not exist: {eval_python}")
    help_text = evaluator_help(eval_python)
    if "No module named" in help_text:
        raise PilotError(f"swebench is unavailable in {eval_python}: {help_text.strip()}")
    ids = [item["instance_id"] for item in manifest]
    common = [
        eval_python,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.dataset,
        "--instance_ids",
        *ids,
        "--max_workers",
        "1",
    ]
    if "--report_dir" not in help_text:
        print("evaluator has no --report_dir; reports will use its default location")

    if not args.skip_gold:
        gold_dir = root / "eval" / "gold_smoke"
        command = common + ["--predictions_path", "gold", "--run_id", f"{args.run_id}_gold"]
        if "--report_dir" in help_text:
            command += ["--report_dir", str(gold_dir)]
        code = run_evaluator_command(args, command, root / "logs" / "gold_smoke.log")
        if code != 0:
            raise PilotError("official SWE-bench gold smoke test failed; see logs/gold_smoke.log")

    for config in CONFIGS:
        eval_dir = root / "eval" / config["tag"]
        command = common + [
            "--predictions_path",
            str(prediction_path(root, config["tag"])),
            "--run_id",
            f"{args.run_id}_{config['tag']}",
        ]
        if "--report_dir" in help_text:
            command += ["--report_dir", str(eval_dir)]
        code = run_evaluator_command(args, command, root / "logs" / f"{config['tag']}.log")
        if code != 0:
            print(f"WARNING: evaluator failed for {config['tag']}; report will record infrastructure failure")


def recursively_find_eval_records(value: Any, path: Path, out: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        instance_id = value.get("instance_id")
        if instance_id and any(key in value for key in ("resolved", "tests_status", "FAIL_TO_PASS", "fail_to_pass")):
            record = dict(value)
            record["evaluation_report_path"] = str(path)
            out.append(record)
        for child in value.values():
            recursively_find_eval_records(child, path, out)
    elif isinstance(value, list):
        for child in value:
            recursively_find_eval_records(child, path, out)


def collect_eval_records(directory: Path) -> dict[str, dict[str, Any]]:
    found: list[dict[str, Any]] = []
    summaries: list[tuple[Path, dict[str, Any]]] = []
    if directory.exists():
        for path in directory.rglob("*.json"):
            try:
                with path.open() as handle:
                    value = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            recursively_find_eval_records(value, path, found)
            if isinstance(value, dict) and "submitted_ids" in value:
                summaries.append((path, value))

    records = {record["instance_id"]: record for record in found}
    # swebench 5.x writes one aggregate report per run rather than one JSON
    # object per instance. Expand that summary so the merged report still has
    # a quality row for every generated instance.
    for path, summary in summaries:
        resolved_ids = set(summary.get("resolved_ids") or [])
        completed_ids = set(summary.get("completed_ids") or [])
        infra_ids = set(summary.get("infra_failure_ids") or [])
        for instance_id in summary.get("submitted_ids") or []:
            records.setdefault(
                instance_id,
                {
                    "instance_id": instance_id,
                    "resolved": instance_id in resolved_ids,
                    "evaluation_completed": instance_id in completed_ids,
                    "infra_failure": instance_id in infra_ids,
                    "evaluation_report_path": str(path),
                },
            )
    return records


def report_results(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    manifest = load_manifest(root)
    merged_path = root / "reports" / "merged.jsonl"
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "dataset": args.dataset,
        "subset_size": len(manifest),
        "configs": {},
    }
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w") as merged:
        for config in CONFIGS:
            telemetry = latest_by_key(read_jsonl(telemetry_path(root, config["tag"])), ("instance_id",))
            eval_records = collect_eval_records(root / "eval" / config["tag"])
            rows = []
            for item in manifest:
                generated = telemetry.get((item["instance_id"],))
                evaluated = eval_records.get(item["instance_id"], {})
                row = {
                    "config": config["tag"],
                    "instance_id": item["instance_id"],
                    "generation": generated,
                    "evaluation": evaluated or None,
                }
                rows.append(row)
                merged.write(json.dumps(row, default=str) + "\n")
            resolved = [row["evaluation"].get("resolved") for row in rows if row["evaluation"] and "resolved" in row["evaluation"]]
            resolved_true = sum(value is True or str(value).lower() == "true" for value in resolved)
            tps = [row["generation"].get("output_tokens_per_s") for row in rows if row["generation"] and row["generation"].get("output_tokens_per_s")]
            energy = [row["generation"].get("energy_joules") for row in rows if row["generation"] and row["generation"].get("energy_joules") is not None]
            patch_valid = sum(row["generation"].get("patch_status") == "model_valid" for row in rows if row["generation"])
            fallback = sum(row["generation"].get("patch_status", "").endswith("fallback") for row in rows if row["generation"])
            summary["configs"][config["tag"]] = {
                "instances": len(rows),
                "resolved": resolved_true,
                "resolved_rate": resolved_true / len(resolved) if resolved else None,
                "evaluated_instances": len(resolved),
                "model_valid_patches": patch_valid,
                "fallback_patches": fallback,
                "mean_output_tokens_per_s": float(np.mean(tps)) if tps else None,
                "mean_energy_joules": float(np.mean(energy)) if energy else None,
                "mean_avg_power_watts": float(np.mean([
                    row["generation"].get("avg_power_watts")
                    for row in rows
                    if row["generation"] and row["generation"].get("avg_power_watts") is not None
                ])) if any(row["generation"] and row["generation"].get("avg_power_watts") is not None for row in rows) else None,
            }
    write_json(root / "reports" / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    print(f"merged report: {merged_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="run", choices=["prepare", "generate", "validate", "evaluate", "report", "run"])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--repo")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction", type=float, default=0.55)
    parser.add_argument("--launch-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--evaluator-python")
    parser.add_argument("--skip-gold", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_manifest(args)
    elif args.command == "generate":
        generate_predictions(args)
    elif args.command == "validate":
        validate_predictions(args)
    elif args.command == "evaluate":
        evaluate_predictions(args)
    elif args.command == "report":
        report_results(args)
    else:
        root = Path(args.root).expanduser().resolve()
        if not manifest_path(root).exists():
            prepare_manifest(args)
        generate_predictions(args)
        validate_predictions(args)
        evaluate_predictions(args)
        report_results(args)


if __name__ == "__main__":
    main()
