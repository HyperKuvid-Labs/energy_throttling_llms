import argparse

import numpy as np

from common import build_numpy_dataset, representative_state, valid_actions_for_bs, print_vs_reference, save_picks

# disjoint linucb (li et al 2010): one ridge regressor per action, updated
# incrementally over the logged rows. matches the actual problem shape --
# 19 actions, 4-dim context, one reward per pull -- way better fit than
# ddpg's continuous actor ever was for a genuinely discrete choice


def train_linucb(states, actions, rewards, n_actions, lam=1.0):
    d = states.shape[1]
    A = [lam * np.eye(d) for _ in range(n_actions)]
    b = [np.zeros(d) for _ in range(n_actions)]
    for x, a, r in zip(states, actions, rewards):
        A[a] += np.outer(x, x)
        b[a] += r * x
    theta = [np.linalg.solve(A[a], b[a]) for a in range(n_actions)]
    return theta, A


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--power-limit", type=float, default=80.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=1.0, help="ucb exploration bonus, only matters for online collection")
    args = p.parse_args()

    space, states, actions, rewards, meta, baselines = build_numpy_dataset(
        args.power_limit, args.energy_weight)
    print(f"{len(meta)} usable rows, {len(space)} discrete actions")

    theta, A = train_linucb(states, actions, rewards, len(space))

    # A is saved alongside theta since it's what you'd need to keep updating this
    # incrementally later instead of refitting from scratch on the whole dataset
    np.savez("linucb_model.npz", theta=np.stack(theta), A=np.stack(A),
             actions=np.array(space.actions))
    print("saved linucb_model.npz")

    # deployment is greedy off the fitted theta -- the ucb bonus exists to pick
    # what to try next during data collection, and the data's already collected,
    # so there's nothing left for it to explore
    batch_sizes = sorted({m["batch_size"] for m in meta})
    picks = {}
    for bs in batch_sizes:
        x = representative_state(meta, bs)
        valid = valid_actions_for_bs(space, meta, bs)
        scores = [theta[a] @ x for a in valid]
        best = valid[int(np.argmax(scores))]
        picks[bs] = space.actions[best]

    print_vs_reference("linucb", meta, picks)
    save_picks("linucb", picks)


if __name__ == "__main__":
    main()
