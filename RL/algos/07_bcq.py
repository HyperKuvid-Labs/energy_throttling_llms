import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# common does the sys.path insert for the parent dir, has to import before policy
from common import build_torch_dataset, representative_state, valid_actions_for_bs, print_vs_reference, save_picks
from policy import ActionSpace, QNetwork

# discrete bcq (fujimoto et al, benchmarking batch deep rl appendix): a behavior
# cloning net learns which actions the logging policy actually picked per state,
# and at decision time we only argmax q over actions the cloning net is
# reasonably confident in -- keeps the policy from betting on an action that
# looks good to the q-net purely because it never saw enough of it to know better.
#
# worth flagging: our logging policy is a uniform sweep over every valid action
# per batch size, not a skewed real-world log, so the cloning net ends up
# predicting roughly uniform probability over the valid set too. bcq's whole
# value proposition (steering away from underrepresented actions) mostly
# collapses to the plain validity mask here -- same situation as the dr script.


class BehaviorNet(nn.Module):
    def __init__(self, state_dims, n_actions, hidden=128, lr=1e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dims, hidden), nn.GELU(),
            nn.Linear(hidden, n_actions),
        )
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        return self.net(state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--power-limit", type=float, default=80.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.3, help="min prob ratio vs the cloning net's argmax to be considered")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    space, states, actions, rewards, meta, baselines = build_torch_dataset(
        args.power_limit, args.energy_weight)
    print(f"{len(meta)} usable rows, {len(space)} discrete actions")

    g = torch.Generator().manual_seed(args.seed)
    n = len(states)

    behavior = BehaviorNet(state_dims=states.shape[1], n_actions=len(space))
    q = QNetwork(state_dims=states.shape[1], n_actions=len(space))
    states_d = states.to(q.device)
    actions_d = actions.to(q.device)
    rewards_d = rewards.to(q.device)

    # behavior cloning: plain classification, predict which action a state got
    for epoch in range(args.epochs):
        perm = torch.randperm(n, generator=g).to(q.device)
        epoch_loss = 0.0
        for i in range(0, n, args.batch_size):
            b = perm[i:i + args.batch_size]
            logits = behavior(states_d[b])
            loss = F.cross_entropy(logits, actions_d[b])
            behavior.optimizer.zero_grad()
            loss.backward()
            behavior.optimizer.step()
            epoch_loss += loss.item() * len(b)
        if (epoch + 1) % 100 == 0:
            print(f"  behavior epoch {epoch + 1:4d}  loss {epoch_loss / n:.5f}")

    # q-network: same fitted-q regression as train_offline.py, trained independently
    for epoch in range(args.epochs):
        perm = torch.randperm(n, generator=g).to(q.device)
        epoch_loss = 0.0
        for i in range(0, n, args.batch_size):
            b = perm[i:i + args.batch_size]
            pred = q(states_d[b]).gather(1, actions_d[b].unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(pred, rewards_d[b])
            q.optimizer.zero_grad()
            loss.backward()
            q.optimizer.step()
            epoch_loss += loss.item() * len(b)
        if (epoch + 1) % 100 == 0:
            print(f"  q epoch {epoch + 1:4d}  loss {epoch_loss / n:.5f}")

    batch_sizes = sorted({m["batch_size"] for m in meta})
    picks = {}
    for bs in batch_sizes:
        x = torch.tensor([representative_state(meta, bs)], dtype=torch.float).to(q.device)
        valid = valid_actions_for_bs(space, meta, bs)

        with torch.no_grad():
            probs = F.softmax(behavior(x), dim=-1).squeeze(0)
            qv = q(x).squeeze(0)

        max_prob = probs[valid].max().item()
        allowed = [a for a in valid if probs[a].item() / max_prob > args.threshold]
        if not allowed:
            allowed = valid  # threshold too strict for this state, fall back to the validity mask
        best = max(allowed, key=lambda a: qv[a].item())
        picks[bs] = space.actions[best]

    print_vs_reference("bcq", meta, picks)
    save_picks("bcq", picks)

    torch.save({"behavior_state_dict": behavior.state_dict(),
                "q_state_dict": q.state_dict(),
                "actions": space.actions}, "bcq_policy.pth")
    print("saved bcq_policy.pth")


if __name__ == "__main__":
    main()
