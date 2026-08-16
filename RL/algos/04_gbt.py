import argparse

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from common import build_numpy_dataset, representative_state, valid_actions_for_bs, print_vs_reference, save_picks

# no xgboost/lightgbm installed on this laptop, sklearn's GradientBoostingRegressor
# does the same job for a dataset this small. feature = state (4 dims) + raw action
# index as a 5th feature, not one-hot -- trees split on it fine, and keeping it out
# of one-hot means feature_importances_ stays readable per raw signal instead of
# smeared across 19 dummy columns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--power-limit", type=float, default=80.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    space, states, actions, rewards, meta, baselines = build_numpy_dataset(
        args.power_limit, args.energy_weight)
    print(f"{len(meta)} usable rows, {len(space)} discrete actions")

    X = np.column_stack([states, actions.astype(np.float64) / len(space)])
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=args.seed)
    model.fit(X, rewards)

    joblib.dump(model, "gbt_model.joblib")
    print("saved gbt_model.joblib")

    names = ["batch_size", "gpu_temp", "gpu_mem", "gpu_util", "action_idx"]
    print("feature importances (does it actually use temp/mem/util or just batch_size):")
    for n, imp in sorted(zip(names, model.feature_importances_), key=lambda t: -t[1]):
        print(f"  {n:<12} {imp:.4f}")

    batch_sizes = sorted({m["batch_size"] for m in meta})
    picks = {}
    for bs in batch_sizes:
        x = representative_state(meta, bs)
        valid = valid_actions_for_bs(space, meta, bs)
        cand = np.column_stack([np.tile(x, (len(valid), 1)),
                                 np.array(valid, dtype=np.float64) / len(space)])
        preds = model.predict(cand)
        best = valid[int(np.argmax(preds))]
        picks[bs] = space.actions[best]

    print_vs_reference("gradient boosted trees", meta, picks)
    save_picks("gbt", picks)


if __name__ == "__main__":
    main()
