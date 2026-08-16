import argparse

import numpy as np

from common import build_numpy_dataset, representative_state, valid_actions_for_bs, print_vs_reference, save_picks

# bayesian linear regression per action, same A/b bookkeeping as linucb but
# instead of an ucb bonus we keep the posterior covariance and sample theta
# from it at decision time. gives a distribution over "how good is this
# config" instead of a point estimate, which is the whole selling point
# over the mlp bandit -- that one just spits out a number with no uncertainty


def fit_posteriors(states, actions, rewards, n_actions, lam=1.0):
    d = states.shape[1]
    A = [lam * np.eye(d) for _ in range(n_actions)]
    b = [np.zeros(d) for _ in range(n_actions)]
    for x, a, r in zip(states, actions, rewards):
        A[a] += np.outer(x, x)
        b[a] += r * x
    mean = [np.linalg.solve(A[a], b[a]) for a in range(n_actions)]
    cov = [np.linalg.inv(A[a]) for a in range(n_actions)]
    return mean, cov


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--power-limit", type=float, default=80.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    space, states, actions, rewards, meta, baselines = build_numpy_dataset(
        args.power_limit, args.energy_weight)
    print(f"{len(meta)} usable rows, {len(space)} discrete actions")

    mean, cov = fit_posteriors(states, actions, rewards, len(space))

    # the posterior itself (mean + covariance per action) is the model -- saving
    # both, not just mean, since a fresh sample at decision time needs cov too
    np.savez("thompson_sampling_model.npz", mean=np.stack(mean), cov=np.stack(cov),
             actions=np.array(space.actions))
    print("saved thompson_sampling_model.npz")

    # one posterior sample per batch size, same as a real thompson sampling
    # deployment would do for that decision -- seeded so the pick is reproducible
    # for the live validation step later, not because ts is normally deterministic
    batch_sizes = sorted({m["batch_size"] for m in meta})
    picks = {}
    for bs in batch_sizes:
        x = representative_state(meta, bs)
        valid = valid_actions_for_bs(space, meta, bs)
        sampled_scores = []
        for a in valid:
            theta_sample = rng.multivariate_normal(mean[a], cov[a])
            sampled_scores.append(theta_sample @ x)
        best = valid[int(np.argmax(sampled_scores))]
        picks[bs] = space.actions[best]

    print_vs_reference("thompson sampling", meta, picks)
    save_picks("thompson_sampling", picks)


if __name__ == "__main__":
    main()
