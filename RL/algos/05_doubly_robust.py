import argparse

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from common import build_numpy_dataset, representative_state, valid_actions_for_bs, print_vs_reference, save_picks

# doubly robust off-policy value estimation, per dudik et al -- combines a direct
# reward model with an importance-weighted correction on top of it.
#
# caveat worth being upfront about: the sweep enumerates every valid action for
# every batch size (not a policy stochastically choosing one action per context),
# so the "behavior policy" here is a modeling assumption (uniform over the valid
# actions for that bs) rather than the literal collection process. with full grid
# coverage like this, ips/dr mostly earns its keep as a sanity check that the
# correction term doesn't blow up the direct model's picks -- the real payoff of
# dr (correcting for unobserved actions) shows up when the logged data is a
# genuine partial sample, which ours isn't.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--power-limit", type=float, default=80.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    args = p.parse_args()

    space, states, actions, rewards, meta, baselines = build_numpy_dataset(
        args.power_limit, args.energy_weight)
    print(f"{len(meta)} usable rows, {len(space)} discrete actions")

    # direct model: plain linear regression over state + action idx. deliberately
    # not the per-cell empirical mean -- a smoothed model is what gives the ips
    # correction term something real to correct, instead of trivially cancelling
    X = np.column_stack([states, actions.astype(np.float64) / len(space)])
    dm = LinearRegression().fit(X, rewards)

    joblib.dump(dm, "doubly_robust_model.joblib")
    print("saved doubly_robust_model.joblib")

    # the correction term below is recomputed from the raw dataset at pick time,
    # not something that gets persisted -- the direct model above is the only
    # piece that's actually reusable without the original rows on hand
    def dm_predict(x, a):
        return dm.predict([[*x, a / len(space)]])[0]

    batch_sizes = sorted({m["batch_size"] for m in meta})
    picks = {}
    diagnostics = {}
    for bs in batch_sizes:
        bs_idx = [i for i, m in enumerate(meta) if m["batch_size"] == bs]
        valid = valid_actions_for_bs(space, meta, bs)
        n_valid = len(valid)
        propensity = 1.0 / n_valid  # uniform over the valid set, per the caveat above

        dr_values = {}
        direct_values = {}
        for a in valid:
            terms = []
            for i in bs_idx:
                x_i, a_i, r_i = states[i], actions[i], rewards[i]
                dm_a = dm_predict(x_i, a)
                correction = (r_i - dm_predict(x_i, a_i)) / propensity if a_i == a else 0.0
                terms.append(dm_a + correction)
            dr_values[a] = float(np.mean(terms))
            direct_values[a] = dm_predict(representative_state(meta, bs), a)

        best_dr = max(dr_values, key=dr_values.get)
        best_direct = max(direct_values, key=direct_values.get)
        picks[bs] = space.actions[best_dr]
        diagnostics[bs] = (best_dr == best_direct)

    print("dr pick matches plain direct-method pick (expected, given full coverage):")
    for bs, matched in diagnostics.items():
        print(f"  bs={bs}  matched={matched}")

    print_vs_reference("doubly robust", meta, picks)
    save_picks("doubly_robust", picks)


if __name__ == "__main__":
    main()
