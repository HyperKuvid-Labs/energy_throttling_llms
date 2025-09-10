import numpy as np
from collections import deque


class RuntimeRewardCalculator:
    def __init__(self, history_len=500):
        self.metric_keys = [
            "throughput_tokens_sec",
            "time_to_first_token_ms",
            "mean_inter_token_time_ms",
            "total_inference_time_ms",
            "energy_per_token_j"
        ]
        self.history = {k: deque(maxlen=history_len) for k in self.metric_keys}

        # Fallback normalization ranges
        self.bounds = {
            "throughput_tokens_sec": (0.1, 10000),
            "time_to_first_token_ms": (0.1, 1000),
            "mean_inter_token_time_ms": (0.1, 500),
            "total_inference_time_ms": (1, 10000),
            "energy_per_token_j": (0.001, 10.0),
        }

        # Priority weights (sum ≈1)
        self.weights = {
            "throughput_tokens_sec": 0.35,         
            "time_to_first_token_ms": 0.2,         
            "mean_inter_token_time_ms": 0.2,       
            "total_inference_time_ms": 0.15,       
            "energy_per_token_j": 0.1,             
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
        # update history
        for k in self.metric_keys:
            v = metrics.get(k)
            if v is not None:
                self.history[k].append(v)

        normed = {}
        for k in self.metric_keys:
            v = metrics.get(k, self.bounds[k][0])
            if k == "throughput_tokens_sec":
                normed[k] = self._zsig(k, v)
            else:
                normed[k] = 1.0 - self._zsig(k, v)

        reward = sum(self.weights[k] * normed[k] for k in self.weights)
        return max(0.0, min(1.0, reward))


_calc = RuntimeRewardCalculator()


def get_runtime_reward(metrics: dict) -> float:
    """
    Call with runtime metrics to get scalar reward [0,1].
    """
    return _calc.compute_reward(metrics)


if __name__ == "__main__":
    dummy_metrics = {
        "throughput_tokens_sec": 500,
        "time_to_first_token_ms": 20,
        "mean_inter_token_time_ms": 2.0,
        "total_inference_time_ms": 1000,
        "energy_per_token_j": 0.02
    }
    print("Runtime Reward:", get_runtime_reward(dummy_metrics))
