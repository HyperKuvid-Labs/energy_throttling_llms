import argparse
import json
from collections import defaultdict

from common import build_numpy_dataset, print_vs_reference, save_picks

# dumbest possible baseline: ignore state entirely, just remember which config
# scored best historically for each batch size. if the mlp bandit can't beat
# this it's not earning its complexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--power-limit", type=float, default=80.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    args = p.parse_args()

    space, states, actions, rewards, meta, baselines = build_numpy_dataset(
        args.power_limit, args.energy_weight)
    print(f"{len(meta)} usable rows, {len(space)} discrete actions")

    # average reward per (bs, config) across repeats first, so one noisy repeat
    # doesn't hijack the pick -- a raw per-row argmax would just be picking noise
    sums = defaultdict(float)
    counts = defaultdict(int)
    for m in meta:
        key = (m["batch_size"], m["steps"], m["topk"], m["num_draft_tokens"])
        sums[key] += m["reward"]
        counts[key] += 1
    mean_reward = {k: sums[k] / counts[k] for k in sums}

    batch_sizes = sorted({m["batch_size"] for m in meta})
    picks = {}
    for bs in batch_sizes:
        bs_keys = [k for k in mean_reward if k[0] == bs]
        best_key = max(bs_keys, key=lambda k: mean_reward[k])
        picks[bs] = best_key[1:]

    print_vs_reference("lookup table", meta, picks)
    save_picks("lookup_table", picks)

    # the mean_reward table is the whole model here, no separate inference step,
    # a lookup just indexes into this by bs at decision time
    model = {str(bs): {"config": list(picks[bs]), "mean_reward": mean_reward[(bs,) + picks[bs]]}
             for bs in batch_sizes}
    with open("lookup_table_model.json", "w") as f:
        json.dump(model, f, indent=2)
    print("saved lookup_table_model.json")


if __name__ == "__main__":
    main()
