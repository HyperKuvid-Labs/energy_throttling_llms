import sys
import os
import json

# all these scripts live one level under RL/, need the parent on path to reuse
# the exact same data loading / reward / action-space code the mlp bandit uses,
# no point reimplementing that per script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from policy import ActionSpace
from reward import compute_reward
from sweep_config import REFERENCE_CONFIG
from train_offline import load_rows, state_features, build_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.dirname(HERE)
DATASET = os.path.join(RL_DIR, "hf_push", "eagle3_energy_sweep.jsonl")
PICKS_PATH = os.path.join(HERE, "results", "picks.json")


def build_numpy_dataset(power_limit=80.0, energy_weight=0.5):
    # same rows/reward as train_offline.build_dataset, just numpy instead of torch --
    # none of the algos in this folder need autograd, sklearn and the manual linalg
    # ones all want plain arrays
    rows = load_rows(DATASET)
    baselines = {r["batch_size"]: r.get("speed_tok_s")
                 for r in rows if r.get("is_baseline") and not r.get("error")}
    space = ActionSpace()
    states, actions, rewards, meta = [], [], [], []
    for row in rows:
        if row.get("error"):
            continue
        idx = space.to_index(row["steps"], row["topk"], row["num_draft_tokens"])
        if idx is None:
            continue
        r, parts = compute_reward(row, baselines.get(row["batch_size"]), power_limit,
                                   energy_weight=energy_weight)
        states.append(state_features(row))
        actions.append(idx)
        rewards.append(r)
        meta.append({**row, **parts, "reward": r})
    return (space,
            np.array(states, dtype=np.float64),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float64),
            meta, baselines)


def build_torch_dataset(power_limit=80.0, energy_weight=0.5):
    # cql/bcq need gradients, so they want the torch tensors train_offline.py
    # already builds -- no reason to duplicate that logic in numpy and convert back
    rows = load_rows(DATASET)
    return build_dataset(rows, power_limit, energy_weight)


def representative_state(meta, bs):
    # state features are read off metrics_after in run_sweep, so this is picking
    # with hindsight state, not a live pre-launch sensor read -- same caveat as
    # eval_live.py / eval_live_seeds.py, not fixing it here either
    rows = [m for m in meta if m["batch_size"] == bs]
    return np.array(state_features(rows[0]), dtype=np.float64)


def valid_actions_for_bs(space, meta, bs):
    # not every action in the space is valid at every batch size (vram limits at
    # bs=16 etc), so restrict to actions that actually appear in the dataset for
    # this bs rather than trusting the full action space mask
    seen = {(m["steps"], m["topk"], m["num_draft_tokens"])
            for m in meta if m["batch_size"] == bs}
    return sorted(space.to_index(*cfg) for cfg in seen)


def print_vs_reference(name, meta, picks):
    # picks: dict bs -> (steps, topk, draft)
    by_key = {(m["batch_size"], m["steps"], m["topk"], m["num_draft_tokens"]): m for m in meta}
    print(f"\n=== {name} vs fixed reference ===")
    for bs, cfg in sorted(picks.items()):
        chosen_row = by_key.get((bs,) + tuple(cfg))
        ref_row = by_key.get((bs,) + REFERENCE_CONFIG)
        rows = [m for m in meta if m["batch_size"] == bs]
        best = max(rows, key=lambda m: m["reward"])

        def fmt(tag, row, cfg_):
            if row is None:
                return f"  {tag:<12} {str(cfg_):<12} (not in dataset)"
            return (f"  {tag:<12} {str(cfg_):<12} reward={row['reward']:.4f} "
                    f"speed={row.get('speed_tok_s')}")

        print(f"bs={bs}")
        print(fmt("policy", chosen_row, tuple(cfg)))
        print(fmt("reference", ref_row, REFERENCE_CONFIG))
        print(fmt("oracle", best, (best["steps"], best["topk"], best["num_draft_tokens"])))


def save_picks(name, picks):
    # keeps every algo's picks in one file so validate_algos.py can dedupe across
    # all of them at the end instead of live-benchmarking the same config N times
    os.makedirs(os.path.dirname(PICKS_PATH), exist_ok=True)
    all_picks = {}
    if os.path.exists(PICKS_PATH):
        with open(PICKS_PATH) as f:
            all_picks = json.load(f)
    all_picks[name] = {str(bs): list(cfg) for bs, cfg in picks.items()}
    with open(PICKS_PATH, "w") as f:
        json.dump(all_picks, f, indent=2)
    print(f"saved picks for {name} -> {PICKS_PATH}")
