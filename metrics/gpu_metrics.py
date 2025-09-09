import pynvml
import cupy as cp
import time

pynvml.nvmlInit()

def collect_gpu_metrics(device_index: int = 0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    
    metrics = {}
    metrics["power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    metrics["utilization_pct"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    metrics["memory_used_mib"] = mem_info.used / (1024**2)
    metrics["memory_util_pct"] = (mem_info.used / mem_info.total) * 100
    metrics["temperature_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    metrics["clock_core_mhz"] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    metrics["clock_mem_mhz"] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    metrics["fan_speed_pct"] = pynvml.nvmlDeviceGetFanSpeed(handle)

    #We need to get this from cuda profiling tool say for now we can get this fro NSIGHT Compute
    metrics["pcie_throughput_mb_s"] = None
    metrics["kernel_time_ms"] = None
    metrics["kernel_launch_latency_ms"] = None
    metrics["cuda_streams_active"] = None
    metrics["sm_occupancy_pct"] = None

    return metrics
