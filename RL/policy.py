"""Discrete policy over the valid speculative-decoding configs.

Replaces the continuous actor in components/{fast_updating_actor,target_actor}.py.
Those emitted three sigmoid-scaled scalars and then called torch.round on them,
which has zero gradient almost everywhere -- backprop through the actor produced
`grad: tensor([[0.]])`, so the actor never learned. A straight-through estimator
would fix the gradient but not the second problem: only 75.4% of the continuous
box (steps[1,32] x topk[1,10] x draft[1,64]) satisfies sglang's constraints, so
a quarter of proposed actions were unrunnable.

Enumerating the legal configs solves both at once -- the network emits logits
over a fixed action list, illegal actions are masked to -inf, and there is
nothing to round.

Framing: the cached sweep gives one reward per (batch_size, config) with no
captured state transition, so this is a contextual bandit, not a sequential MDP.
Q-learning here is one-step (gamma = 0). Calling it DDPG would be dressing up a
bandit; the discount only becomes meaningful once configs are switched
mid-session and one choice affects the next state.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from sweep_config import build_grid, is_valid, canonicalize


class ActionSpace:
    """Fixed enumeration of (steps, topk, num_draft_tokens) the policy may pick."""

    def __init__(self, batch_sizes=None):
        # Configs are shared across batch sizes; batch size is a state feature,
        # not part of the action, so dedupe on the config triple alone.
        seen, actions = set(), []
        for _bs, s, k, d in build_grid(batch_sizes=batch_sizes):
            if (s, k, d) not in seen:
                seen.add((s, k, d))
                actions.append((s, k, d))
        self.actions = actions
        self.index = {a: i for i, a in enumerate(actions)}

    def __len__(self):
        return len(self.actions)

    def to_index(self, steps, topk, num_draft_tokens):
        return self.index.get(canonicalize(steps, topk, num_draft_tokens))

    def mask(self, max_steps=None):
        """Boolean legality mask, optionally capping depth for tight VRAM."""
        return torch.tensor(
            [is_valid(*a) and (max_steps is None or a[0] <= max_steps) for a in self.actions],
            dtype=torch.bool,
        )


class QNetwork(nn.Module):
    """Q(state) -> one value per discrete action.

    Emitting all action-values at once (rather than the old Q(state, action)
    scoring one action per forward pass) is what makes masking and exact greedy
    selection cheap -- argmax over the legal subset is a single forward.
    """

    def __init__(self, state_dims, n_actions, hidden=128, lr=1e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dims, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_actions),
        )
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        return self.net(state)

    def q_masked(self, state, mask):
        q = self.forward(state)
        return q.masked_fill(~mask.to(q.device), float("-inf"))

    def act(self, state, mask, epsilon=0.0):
        """Epsilon-greedy over legal actions only."""
        legal = torch.nonzero(mask, as_tuple=False).flatten()
        if torch.rand(1).item() < epsilon:
            return int(legal[torch.randint(len(legal), (1,))].item())
        with torch.no_grad():
            return int(self.q_masked(state, mask).argmax(dim=-1).item())
