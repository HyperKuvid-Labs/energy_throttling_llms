# cpu_reward.py
import psutil
import numpy as np
from collections import deque

class CPURewardCalculator:
    def __init__(self,
                 history_len=500,
                 cpu_target_pct=70.0,
                 penalty_multiplier=3.0):
        """
        CPU reward calculator with rolling normalization and penalties.
        """
        self.history = {k: deque(maxlen=history_len) for k in [
            "cpu_utilization_pct", "process_cpu_time_s",
            "system_ram_used_mib", "swap_usage_mib",
            "page_faults", "disk_io_mb_s", "net_io_mb_s"
        ]}
        self.cpu_target_pct = cpu_target_pct
        self.penalty_multiplier = penalty_multiplier

        # Fallback bounds
        self.bounds = {
            "cpu_utilization_pct": (0, 100),
            "process_cpu_time_s": (0, 10),
            "system_ram_used_mib": (0, psutil.virtual_memory().total / (1024**2)),
            "swap_usage_mib": (0, psutil.swap_memory().total / (1024**2)),
            "page_faults": (0, 1e6),
            "disk_io_mb_s": (0, 1000),
            "net_io_mb_s": (0, 1000),
        }

        # Importance weights (sum ~ 1)
        self.weights = {
            "cpu_utilization_pct": 0.25,
            "system_ram_used_mib": 0.2,
            "swap_usage_mib": 0.2,
            "page_faults": 0.15,
            "disk_io_mb_s": 0.1,
            "net_io_mb_s": 0.05,
            "process_cpu_time_s": 0.05,
        }

    def _zsig(self, key, value):
        """z-score → sigmoid normalization with min-max fallback"""
        hist = self.history[key]
        if len(hist) > 20:  # enough samples
            mean = np.mean(hist)
            std = np.std(hist) or 1.0
            z = (value - mean) / std
            return 1 / (1 + np.exp(-z))
        else:
            lo, hi = self.bounds[key]
            return max(0.0, min(1.0, (value - lo) / (hi - lo + 1e-9)))

    def _cpu_util_score(self, util):
        """Gaussian utility centered on target CPU utilization"""
        sigma = 15.0
        return float(np.exp(-0.5 * ((util - self.cpu_target_pct) / sigma) ** 2))

    def compute_reward(self):
        metrics = {
            "cpu_utilization_pct": psutil.cpu_percent(interval=0.1),
            "process_cpu_time_s": psutil.Process().cpu_times().user,
            "system_ram_used_mib": psutil.virtual_memory().used / (1024**2),
            "swap_usage_mib": psutil.swap_memory().used / (1024**2),
            "page_faults": psutil.swap_memory().sin,
            "disk_io_mb_s": psutil.disk_io_counters().read_bytes / (1024**2),
            "net_io_mb_s": psutil.net_io_counters().bytes_sent / (1024**2),
        }

        for k, v in metrics.items():
            self.history[k].append(v)

        normed = {}
        for k, v in metrics.items():
            if k == "cpu_utilization_pct":
                normed[k] = self._cpu_util_score(v)
            else:
                normed[k] = self._zsig(k, v)

        weighted_sum = sum(self.weights[k] * normed[k] for k in self.weights)

        # Hard penalty for swap/page faults
        penalty = 0
        if metrics["swap_usage_mib"] > 0 or metrics["page_faults"] > 1000:
            penalty = self.penalty_multiplier * (normed["swap_usage_mib"] + normed["page_faults"])
            weighted_sum -= penalty

        # Final reward [0,1]
        reward = max(0.0, min(1.0, weighted_sum))
        return reward


# Global calculator instance
_calc = CPURewardCalculator()

def get_cpu_reward():
    """
    Call this anywhere in your project to get a scalar CPU reward [0,1].
    """
    return _calc.compute_reward()


if __name__ == "__main__":
    print("CPU Reward:", get_cpu_reward())
