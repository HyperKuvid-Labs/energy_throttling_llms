# gpu_reward.py
import numpy as np
from collections import deque
import pynvml

pynvml.nvmlInit()


class GPURewardCalculator:
    def __init__(self, history_len=500, device_index=0):
        """
        Reward calculator for GPU metrics (NVIDIA NVML + optional CUDA profiling hooks).
        """
        self.device_index = device_index

        self.metric_keys = [
            "power_w", "utilization_pct", "memory_util_pct",
            "temperature_c", "clock_core_mhz", "clock_mem_mhz",
            "fan_speed_pct",
            # extended optional fields
            "pcie_throughput_mb_s", "kernel_time_ms",
            "kernel_launch_latency_ms", "cuda_streams_active", "sm_occupancy_pct"
        ]

        self.history = {k: deque(maxlen=history_len) for k in self.metric_keys}

        # Fallback normalization ranges
        self.bounds = {
            "power_w": (50, 300),                   # W
            "utilization_pct": (0, 100),            # %
            "memory_util_pct": (0, 100),            # %
            "temperature_c": (30, 95),              # C
            "clock_core_mhz": (300, 2500),          # MHz
            "clock_mem_mhz": (300, 6000),           # MHz
            "fan_speed_pct": (0, 100),              # %
            "pcie_throughput_mb_s": (0, 16000),     # PCIe 4.0 x16 ~ 16 GB/s
            "kernel_time_ms": (0.01, 20.0),         # kernel duration per batch
            "kernel_launch_latency_ms": (0.01, 5.0),
            "cuda_streams_active": (1, 32),
            "sm_occupancy_pct": (0, 100),
        }

        # Priority weights (sum ≈ 1)
        self.weights = {
            "utilization_pct": 0.25,          # want good utilization
            "memory_util_pct": 0.15,          # efficient memory use
            "power_w": 0.1,                   # efficiency
            "temperature_c": 0.1,             # avoid throttling
            "fan_speed_pct": 0.05,            # quieter = better
            "clock_core_mhz": 0.05,           # faster = better
            "clock_mem_mhz": 0.05,
            "pcie_throughput_mb_s": 0.1,      # higher throughput = better
            "kernel_time_ms": 0.05,           # lower = better
            "kernel_launch_latency_ms": 0.05, # lower = better
            "cuda_streams_active": 0.025,     # moderate = better
            "sm_occupancy_pct": 0.025,        # higher = better
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

    def compute_reward(self):
        """
        Collect GPU stats and compute a normalized reward [0,1].
        """
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        metrics = {
            "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
            "utilization_pct": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "memory_util_pct": (mem_info.used / mem_info.total) * 100,
            "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            "clock_core_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            "clock_mem_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
            "fan_speed_pct": pynvml.nvmlDeviceGetFanSpeed(handle),
            # I have to get these from nvidia hooks or profiling like NSIGHT COMPUTE
            "pcie_throughput_mb_s": 8000.0,       
            "kernel_time_ms": 2.0,                 
            "kernel_launch_latency_ms": 0.5,
            "cuda_streams_active": 4,
            "sm_occupancy_pct": 75.0,
        }

        for k, v in metrics.items():
            self.history[k].append(v)

        normed = {}
        for k, v in metrics.items():
            if k in ["utilization_pct", "memory_util_pct", "clock_core_mhz", "clock_mem_mhz",
                     "pcie_throughput_mb_s", "cuda_streams_active", "sm_occupancy_pct"]:
                normed[k] = self._zsig(k, v)  # higher = better
            else:
                normed[k] = 1.0 - self._zsig(k, v)  # lower = better

        reward = sum(self.weights[k] * normed[k] for k in self.weights)
        return max(0.0, min(1.0, reward))


_calc = GPURewardCalculator()

def get_gpu_reward():
    """
    Returns scalar GPU reward in [0,1].
    """
    return _calc.compute_reward()


if __name__ == "__main__":
    print("GPU Reward:", get_gpu_reward())
