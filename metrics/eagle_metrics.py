# eagle_reward.py
import numpy as np
from collections import deque

class EAGLERewardCalculator:
    def __init__(self, history_len=500):
        """
        Reward calculator for speculative decoding (EAGLE).
        """
        self.history = {k: deque(maxlen=history_len) for k in [
            "spec_branches", "spec_accept_rate_pct", "rejection_cascade_depth"
        ]}

        # Fallback min/max bounds
        self.bounds = {
            "spec_branches": (0, 50),               # speculative branches
            "spec_accept_rate_pct": (0, 100),       # %
            "rejection_cascade_depth": (0, 20),     # cascades
        }

        # Importance weights (sum ~ 1)
        self.weights = {
            "spec_accept_rate_pct": 0.5,   # main success signal
            "spec_branches": 0.25,         # efficiency
            "rejection_cascade_depth": 0.25,
        }

    def _zsig(self, key, value):
        """z-score → sigmoid with min-max fallback"""
        hist = self.history[key]
        if len(hist) > 20:
            mean = np.mean(hist)
            std = np.std(hist) or 1.0
            z = (value - mean) / std
            return 1 / (1 + np.exp(-z))
        else:
            lo, hi = self.bounds[key]
            return max(0.0, min(1.0, (value - lo) / (hi - lo + 1e-9)))

    def compute_reward(self, stats: dict):
        """
        stats: dict from SGLang hooks with keys:
          - branches
          - accepted
          - attempted
          - cascades
        """
        metrics = {}
        metrics["spec_branches"] = stats.get("branches", 0)
        metrics["spec_accept_rate_pct"] = (stats.get("accepted", 0) / max(1, stats.get("attempted", 1))) * 100
        metrics["rejection_cascade_depth"] = stats.get("cascades", 0)

        # Update history
        for k, v in metrics.items():
            self.history[k].append(v)

        # Normalize
        normed = {}
        for k, v in metrics.items():
            if k == "spec_accept_rate_pct":
                # Higher is better
                normed[k] = self._zsig(k, v)
            else:
                # Lower is better → invert after normalization
                normed[k] = 1.0 - self._zsig(k, v)

        # Weighted sum
        reward = sum(self.weights[k] * normed[k] for k in self.weights)
        return max(0.0, min(1.0, reward))


# Global calculator
_calc = EAGLERewardCalculator()

def get_eagle_reward(stats: dict):
    """
    Call this with speculative decoding stats to get scalar reward [0,1].
    """
    return _calc.compute_reward(stats)


if __name__ == "__main__":
    test_stats = {"branches": 10, "accepted": 80, "attempted": 100, "cascades": 2}
    print("EAGLE Reward:", get_eagle_reward(test_stats))
