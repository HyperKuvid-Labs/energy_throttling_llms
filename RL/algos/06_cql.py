import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

# common does the sys.path insert for the parent dir, has to import before policy
from common import build_torch_dataset, representative_state, valid_actions_for_bs, print_vs_reference, save_picks
from policy import ActionSpace, QNetwork

# conservative q-learning: same fitted-q setup as train_offline.py, plus a
# penalty that pushes down q-values for actions the data never confirms and
# pushes up the one actually observed. point of this is guarding against the
# mlp bandit's real weakness -- only 3 samples per action, so a plain regression
# can get confidently wrong about an action it barely saw. cql trades some of
# that confidence away on purpose.


def cql_loss(q_values, actions, rewards, alpha):
    pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    td_loss = F.mse_loss(pred, rewards)
    # logsumexp over all actions approximates "how much probability mass would
    # a softmax policy put on actions we're not sure about" -- pushing this down
    # while pushing up the observed action's q is the actual cql regularizer
    conservative = (torch.logsumexp(q_values, dim=1) - pred).mean()
    return td_loss + alpha * conservative, td_loss, conservative


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--power-limit", type=float, default=80.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--alpha", type=float, default=1.0, help="weight on the conservative penalty")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    space, states, actions, rewards, meta, baselines = build_torch_dataset(
        args.power_limit, args.energy_weight)
    print(f"{len(meta)} usable rows, {len(space)} discrete actions")

    q = QNetwork(state_dims=states.shape[1], n_actions=len(space))
    states, actions, rewards = states.to(q.device), actions.to(q.device), rewards.to(q.device)

    g = torch.Generator().manual_seed(args.seed)
    n = len(states)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, generator=g).to(q.device)
        epoch_loss = epoch_td = epoch_cons = 0.0
        for i in range(0, n, args.batch_size):
            b = perm[i:i + args.batch_size]
            qv = q(states[b])
            loss, td, cons = cql_loss(qv, actions[b], rewards[b], args.alpha)
            q.optimizer.zero_grad()
            loss.backward()
            q.optimizer.step()
            epoch_loss += loss.item() * len(b)
            epoch_td += td.item() * len(b)
            epoch_cons += cons.item() * len(b)
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch + 1:4d}  loss {epoch_loss / n:.5f}  "
                  f"td {epoch_td / n:.5f}  conservative {epoch_cons / n:.5f}")

    batch_sizes = sorted({m["batch_size"] for m in meta})
    picks = {}
    for bs in batch_sizes:
        x = torch.tensor([representative_state(meta, bs)], dtype=torch.float).to(q.device)
        valid = valid_actions_for_bs(space, meta, bs)
        mask = torch.zeros(len(space), dtype=torch.bool)
        mask[valid] = True
        chosen = space.actions[q.act(x, mask, epsilon=0.0)]
        picks[bs] = chosen

    print_vs_reference("cql", meta, picks)
    save_picks("cql", picks)

    torch.save({"state_dict": q.state_dict(), "actions": space.actions}, "cql_policy.pth")
    print("saved cql_policy.pth")


if __name__ == "__main__":
    main()
