import json

RL_DIR = "/home/pradheep/Documents/energy_throttling_llms/RL"
SWEEP_PATH = f"{RL_DIR}/hf_push/eagle3_energy_sweep.jsonl"

TB_ROOT = "/tmp/terminalbench-pilot-20260823"
TB_MERGED = f"{TB_ROOT}/reports/merged.jsonl"

SB_ROOT = "/tmp/swebench-lite-pilot-20260823"
SB_MERGED = f"{SB_ROOT}/reports/merged.jsonl"
SB_CONFIG_SUMMARIES = f"{SB_ROOT}/telemetry/config_summaries.jsonl"

ALL_KEYS = [
    "hardware", "index", "repeat", "batch_size", "steps", "topk", "num_draft_tokens",
    "is_baseline", "error",
    "accept_length", "speed_tok_s", "speed_source", "step_time_s",
    "wall_clock_s", "wall_clock_tok_s", "total_output_tokens",
    "energy_joules", "avg_power_watts", "joules_per_token",
    "gpu_temp_c_before", "gpu_temp_c", "gpu_power_w", "gpu_util_pct",
    "gpu_mem_used_mb", "gpu_throttling", "retry_mem_fraction",
]

CONFIG_SPEC = {
    "no_spec": (0, 0, 0),
    "chosen": (3, 4, 8),
    "chosen_bs16": (3, 2, 4),
}

# These rows measure agentic wall-clock throughput of a multi-turn tool-use agent
# (Terminus 2 / SWE-bench patch generation) -- dominated by tool execution, JSON-parse
# retries, and (for chosen_bs16 on Terminal-Bench) a runaway-generation hang -- not the
# controlled synthetic decode throughput the rest of this dataset measures. They are
# kept as real, unmodified measurements for reference, but flagged via a non-null
# "error" so the existing bandit/offline-RL training code (RL/train_offline.py,
# RL/algos/common.py), which already skips any row with a truthy "error", excludes
# them from reward training automatically. They must also never be marked
# is_baseline=True: that field feeds a per-batch_size dict used to normalize every
# row's throughput_score, and these agentic measurements are not comparable
# baselines for the synthetic sweep grid.
EXCLUSION_NOTE = (
    "agentic_pilot_trace: agentic wall-clock measurement (tool-use/patch-gen "
    "overhead included), not a synthetic-sweep measurement -- see speed_source; "
    "intentionally excluded from is_baseline/reward training"
)


def load_jsonl(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_terminalbench_rows(start_index):
    rows = []
    for i, r in enumerate(load_jsonl(TB_MERGED)):
        steps, topk, draft = CONFIG_SPEC[r["config"]]
        gpu_stats = r.get("gpu_stats") or {}
        energy_joules = (gpu_stats["energy_delta_mj"] / 1000.0) if gpu_stats.get("energy_delta_mj") is not None else None
        total_output_tokens = r.get("n_output_tokens")
        joules_per_token = (
            energy_joules / total_output_tokens
            if energy_joules is not None and total_output_tokens
            else None
        )
        row = {k: None for k in ALL_KEYS}
        row.update({
            "hardware": "RTX4060_Laptop",
            "index": start_index + i,
            "repeat": None,
            "batch_size": 1,  # actual concurrency (Harbor runs 1 trial at a time); server_batch_size was 16
            "steps": steps, "topk": topk, "num_draft_tokens": draft,
            "is_baseline": False,
            "error": EXCLUSION_NOTE,
            "accept_length": r.get("avg_spec_accept_length"),
            "speed_tok_s": r.get("tokens_per_second"),
            "speed_source": "terminalbench_pilot",
            "step_time_s": None,
            "wall_clock_s": r.get("agent_execution_duration_s"),
            "wall_clock_tok_s": r.get("tokens_per_second"),
            "total_output_tokens": total_output_tokens,
            "energy_joules": round(energy_joules, 3) if energy_joules is not None else None,
            "avg_power_watts": gpu_stats.get("avg_power_watts"),
            "joules_per_token": round(joules_per_token, 5) if joules_per_token is not None else None,
            "gpu_temp_c_before": None,  # continuous sampler segmented by trial window; no distinct "before" snapshot
            "gpu_temp_c": gpu_stats.get("avg_gpu_temperature_celsius"),
            "gpu_power_w": gpu_stats.get("avg_power_watts"),
            "gpu_util_pct": gpu_stats.get("avg_gpu_utilization_percent"),
            "gpu_mem_used_mb": None,  # not captured by this pilot's segment_gpu_stats helper
            "gpu_throttling": None,
            "retry_mem_fraction": None,
        })
        rows.append(row)
    return rows


def build_swebench_rows(start_index):
    accept_by_config = {r["config"]: r.get("avg_spec_accept_length") for r in load_jsonl(SB_CONFIG_SUMMARIES)}
    rows = []
    for i, r in enumerate(load_jsonl(SB_MERGED)):
        g = r["generation"]
        gb = g.get("gpu_metrics_before") or {}
        ga = g.get("gpu_metrics_after") or {}
        row = {k: None for k in ALL_KEYS}
        row.update({
            "hardware": "RTX4060_Laptop",
            "index": start_index + i,
            "repeat": None,
            "batch_size": 1,  # actual concurrency (effective_concurrency); server_batch_size was 16
            "steps": g.get("steps"), "topk": g.get("topk"), "num_draft_tokens": g.get("num_draft_tokens"),
            "is_baseline": False,
            "error": EXCLUSION_NOTE,
            "accept_length": accept_by_config.get(r["config"]),
            "speed_tok_s": g.get("output_tokens_per_s"),
            "speed_source": "swebench_pilot",
            "step_time_s": None,
            "wall_clock_s": g.get("model_elapsed_s"),
            "wall_clock_tok_s": g.get("output_tokens_per_s"),
            "total_output_tokens": g.get("completion_tokens"),
            "energy_joules": g.get("energy_joules"),
            "avg_power_watts": g.get("avg_power_watts"),
            "joules_per_token": g.get("joules_per_output_token"),
            "gpu_temp_c_before": gb.get("gpu_temperature_celsius"),
            "gpu_temp_c": ga.get("gpu_temperature_celsius"),
            "gpu_power_w": ga.get("gpu_power_watts"),
            "gpu_util_pct": ga.get("gpu_utilization_percent"),
            "gpu_mem_used_mb": ga.get("gpu_memory_used_mb"),
            "gpu_throttling": ga.get("gpu_throttling_active"),
            "retry_mem_fraction": None,
        })
        rows.append(row)
    return rows


def main():
    existing = load_jsonl(SWEEP_PATH)
    next_index = max(r["index"] for r in existing) + 1

    tb_rows = build_terminalbench_rows(next_index)
    sb_rows = build_swebench_rows(next_index + len(tb_rows))

    combined = existing + tb_rows + sb_rows
    with open(SWEEP_PATH, "w") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")

    print(f"existing rows: {len(existing)}")
    print(f"added terminalbench_pilot rows: {len(tb_rows)}")
    print(f"added swebench_pilot rows: {len(sb_rows)}")
    print(f"total rows: {len(combined)}")


if __name__ == "__main__":
    main()
