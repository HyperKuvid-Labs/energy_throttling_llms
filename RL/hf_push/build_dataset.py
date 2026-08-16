import json

RL_DIR = "/home/pradheep/Documents/energy_throttling_llms/RL"
LAPTOP = f"{RL_DIR}/laptop_sweep.jsonl"
RETRY_FILES = [f"{RL_DIR}/bs16_retry.jsonl", f"{RL_DIR}/bs8_retry.jsonl"]
OUT = f"{RL_DIR}/hf_push/eagle3_energy_sweep.jsonl"

ALL_KEYS = [
    "hardware", "index", "repeat", "batch_size", "steps", "topk", "num_draft_tokens",
    "is_baseline", "error",
    "accept_length", "speed_tok_s", "speed_source", "step_time_s",
    "wall_clock_s", "wall_clock_tok_s", "total_output_tokens",
    "energy_joules", "avg_power_watts", "joules_per_token",
    "gpu_temp_c_before", "gpu_temp_c", "gpu_power_w", "gpu_util_pct",
    "gpu_mem_used_mb", "gpu_throttling", "retry_mem_fraction",
]

def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

laptop = load(LAPTOP)

# Retries were run at a lower --mem-fraction-static for configs that failed to
# launch in the main sweep; they carry the same "index" as the row they
# replace, so override by index rather than appending duplicates.
overrides = {}
for path in RETRY_FILES:
    try:
        for r in load(path):
            overrides[r["index"]] = r
    except FileNotFoundError:
        pass

combined = []
n_overridden = 0
for r in laptop:
    if r["index"] in overrides:
        r = overrides[r["index"]]
        n_overridden += 1
    row = {k: r.get(k, None) for k in ALL_KEYS}
    row["hardware"] = "RTX4060_Laptop"
    combined.append(row)

with open(OUT, "w") as f:
    for row in combined:
        f.write(json.dumps(row) + "\n")

n_err = sum(1 for r in combined if r.get("error") is not None)
print(f"total rows: {len(combined)}  (overridden by retry: {n_overridden})")
print(f"error rows: {n_err}")
print(f"success rows: {len(combined) - n_err}")
