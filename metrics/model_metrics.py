# model_reward.py
import numpy as np
from collections import deque


class ModelRewardCalculator:
    def __init__(self, history_len=500):
        self.metric_keys = [
            "prompt_length_tokens",
            "prompt_entropy",
            "queue_length",
            "rolling_avg_latency_ms",
            "p95_latency_ms",
        ]
        self.history = {k: deque(maxlen=history_len) for k in self.metric_keys}

        # Fallback normalization ranges
        self.bounds = {
            "prompt_length_tokens": (1, 2048),
            "prompt_entropy": (0.1, 10.0),
            "queue_length": (0, 64),
            "rolling_avg_latency_ms": (10, 5000),
            "p95_latency_ms": (10, 10000),
        }

        # Weights (sum ≈ 1, priority to latency + queue load)
        self.weights = {
            "prompt_length_tokens": 0.05,
            "prompt_entropy": 0.15,          # balanced entropy is good
            "queue_length": 0.25,            # low queue → higher reward
            "rolling_avg_latency_ms": 0.25,  # fast average latency
            "p95_latency_ms": 0.30,          # stable tail latency
        }

    def _zsig(self, key, value):
        """z-score → sigmoid, fallback min-max"""
        hist = self.history[key]
        if len(hist) > 20:
            mean = np.mean(hist)
            std = np.std(hist) or 1.0
            z = (value - mean) / std
            return 1 / (1 + np.exp(-z))
        else:
            lo, hi = self.bounds[key]
            return max(0.0, min(1.0, (value - lo) / (hi - lo + 1e-9)))

    def compute_reward(self, metrics: dict):
        """
        Takes model-level metrics and returns scalar reward in [0,1].
        """
        # update history
        for k in self.metric_keys:
            self.history[k].append(metrics[k])

        normed = {}
        for k in self.metric_keys:
            if k in ["prompt_length_tokens", "prompt_entropy"]:
                normed[k] = self._zsig(k, metrics[k])  # higher = better
            else:
                normed[k] = 1.0 - self._zsig(k, metrics[k])  # lower = better

        # weighted sum
        reward = sum(self.weights[k] * normed[k] for k in self.metric_keys)
        return max(0.0, min(1.0, reward))


# Global instance
_calc = ModelRewardCalculator()


def get_model_reward(metrics: dict) -> float:
    """
    Wrapper to directly get scalar reward from model metrics.
    """
    return _calc.compute_reward(metrics)


if __name__ == "__main__":
    dummy_metrics = {
        "prompt_length_tokens": 120,
        "prompt_entropy": 3.2,
        "queue_length": 4,
        "rolling_avg_latency_ms": 320,
        "p95_latency_ms": 1200,
    }
    print("Model Reward:", get_model_reward(dummy_metrics))
