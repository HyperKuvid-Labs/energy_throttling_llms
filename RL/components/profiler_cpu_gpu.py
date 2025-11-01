import torch
import pynvml
import psutil
import numpy as np
import json
import os
from datetime import datetime
from functools import wraps


try:
  from zeus.monitor import ZeusMonitor
  za = True
except Exception as e:
  print(f"zeus import exception: {e}")
  za = False


class HardwareMetricsProfiler:
  def __init__(self, gpu_index=0, output_dir="profiled_metrics"):
    self.gpu_index = gpu_index
    self.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    pynvml.nvmlInit()
    self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    if za:
      self.zeus_monitor = ZeusMonitor(gpu_indices=[gpu_index])

    try:
      self.gpu_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle) / 1000.0
    except Exception as e:
      print(f"gpu power limit fetch failed: {e}")
      self.gpu_power_limit = 300.0

    try:
      self.gpu_props = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
      self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
      self.gpu_compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(self.gpu_handle)
      self.gpu_sm_count = pynvml.nvmlDeviceGetNumGpuCores(self.gpu_handle)
    except Exception as e:
      print(f"gpu device props fetch failed: {e}")
      self.gpu_name = "unknown"
      self.gpu_compute_capability = (0, 0)
      self.gpu_sm_count = 0

    self.baselines = self.compute_baseline_metrics()

  def compute_baseline_metrics(self):
    baselines = {}

    try:
      baselines['gpu_temp_max'] = 85.0
      baselines['cpu_temp_max'] = 95.0
      baselines['gpu_power_max'] = self.gpu_power_limit

      baselines['gpu_sm_count'] = self.gpu_sm_count
      baselines['gpu_compute_capability'] = self.gpu_compute_capability
      baselines['gpu_name'] = self.gpu_name

      try:
        max_clocks = pynvml.nvmlDeviceGetMaxClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_SM)
        baselines['gpu_sm_clock_max_mhz'] = max_clocks
      except:
        baselines['gpu_sm_clock_max_mhz'] = 2000.0

      try:
        max_mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_MEM)
        baselines['gpu_memory_clock_max_mhz'] = max_mem_clock
      except:
        baselines['gpu_memory_clock_max_mhz'] = 8000.0

      try:
        pcie_link_max_speed = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(self.gpu_handle)
        pcie_link_max_width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(self.gpu_handle)
        baselines['pcie_max_gen'] = pcie_link_max_speed
        baselines['pcie_max_width'] = pcie_link_max_width
      except:
        baselines['pcie_max_gen'] = 4
        baselines['pcie_max_width'] = 16

      mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
      baselines['gpu_memory_total_mb'] = mem_info.total / (1024 * 1024)

      cpu_freq = psutil.cpu_freq()
      baselines['cpu_frequency_max_mhz'] = cpu_freq.max if cpu_freq else 3000.0
      baselines['cpu_core_count'] = psutil.cpu_count(logical=False)
      baselines['cpu_thread_count'] = psutil.cpu_count(logical=True)

      mem = psutil.virtual_memory()
      baselines['system_memory_total_mb'] = mem.total / (1024 * 1024)

      try:
        l3_cache = 0
        with open('/sys/devices/system/cpu/cpu0/cache/index3/size', 'r') as f:
          cache_str = f.read().strip()
          if 'K' in cache_str:
            l3_cache = int(cache_str.replace('K', '')) / 1024
          elif 'M' in cache_str:
            l3_cache = int(cache_str.replace('M', ''))
        baselines['cpu_l3_cache_mb'] = l3_cache
      except:
        baselines['cpu_l3_cache_mb'] = 0

    except Exception as e:
      print(f"baseline computation failed: {e}")

    return baselines

  def collect_metrics(self):
    metrics = {}

    try:
      metrics['gpu_temperature_celsius'] = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
      metrics['gpu_power_watts'] = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0

      utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
      metrics['gpu_utilization_percent'] = utilization.gpu
      metrics['gpu_memory_utilization_percent'] = utilization.memory

      try:
        metrics['gpu_fan_speed_percent'] = pynvml.nvmlDeviceGetFanSpeed(self.gpu_handle)
      except:
        metrics['gpu_fan_speed_percent'] = 0

      mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
      metrics['gpu_memory_used_mb'] = mem_info.used / (1024 * 1024)
      metrics['gpu_memory_total_mb'] = mem_info.total / (1024 * 1024)
      metrics['gpu_memory_free_mb'] = mem_info.free / (1024 * 1024)

      try:
        metrics['gpu_total_energy_consumed_mj'] = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)
      except:
        metrics['gpu_total_energy_consumed_mj'] = 0

      try:
        metrics['gpu_graphics_clock_mhz'] = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
        metrics['gpu_sm_clock_mhz'] = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_SM)
        metrics['gpu_memory_clock_mhz'] = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_MEM)
      except:
        metrics['gpu_graphics_clock_mhz'] = 0
        metrics['gpu_sm_clock_mhz'] = 0
        metrics['gpu_memory_clock_mhz'] = 0

      try:
        pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(self.gpu_handle)
        pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(self.gpu_handle)
        metrics['pcie_current_gen'] = pcie_gen
        metrics['pcie_current_width'] = pcie_width

        pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(self.gpu_handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
        pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(self.gpu_handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
        metrics['pcie_tx_throughput_kbps'] = pcie_tx
        metrics['pcie_rx_throughput_kbps'] = pcie_rx
      except:
        metrics['pcie_current_gen'] = 0
        metrics['pcie_current_width'] = 0
        metrics['pcie_tx_throughput_kbps'] = 0
        metrics['pcie_rx_throughput_kbps'] = 0

      try:
        encoder_util = pynvml.nvmlDeviceGetEncoderUtilization(self.gpu_handle)
        decoder_util = pynvml.nvmlDeviceGetDecoderUtilization(self.gpu_handle)
        metrics['gpu_encoder_utilization_percent'] = encoder_util[0]
        metrics['gpu_decoder_utilization_percent'] = decoder_util[0]
      except:
        metrics['gpu_encoder_utilization_percent'] = 0
        metrics['gpu_decoder_utilization_percent'] = 0

      try:
        bar1_mem = pynvml.nvmlDeviceGetBAR1MemoryInfo(self.gpu_handle)
        metrics['gpu_bar1_memory_used_mb'] = bar1_mem.bar1Used / (1024 * 1024)
        metrics['gpu_bar1_memory_total_mb'] = bar1_mem.bar1Total / (1024 * 1024)
      except:
        metrics['gpu_bar1_memory_used_mb'] = 0
        metrics['gpu_bar1_memory_total_mb'] = 0

      try:
        perf_state = pynvml.nvmlDeviceGetPerformanceState(self.gpu_handle)
        metrics['gpu_performance_state'] = perf_state
      except:
        metrics['gpu_performance_state'] = 0

      try:
        throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.gpu_handle)
        metrics['gpu_throttle_reasons'] = throttle_reasons
        metrics['gpu_throttling_active'] = throttle_reasons != 0
      except:
        metrics['gpu_throttle_reasons'] = 0
        metrics['gpu_throttling_active'] = False

    except Exception as e:
      print(f"gpu metrics collection failed: {e}")
      metrics['gpu_error'] = str(e)

    try:
      temps = psutil.sensors_temperatures()
      if 'coretemp' in temps:
        cpu_temps = [t.current for t in temps['coretemp'] if t.label.startswith('Core')]
        if cpu_temps:
          metrics['cpu_temperature_celsius_avg'] = np.mean(cpu_temps)
          metrics['cpu_temperature_celsius_max'] = np.max(cpu_temps)
          metrics['cpu_temperature_celsius_per_core'] = cpu_temps
      else:
        all_temps = []
        for sensor_name, sensor_list in temps.items():
          all_temps.extend([t.current for t in sensor_list])
        if all_temps:
          metrics['cpu_temperature_celsius_avg'] = np.mean(all_temps)
          metrics['cpu_temperature_celsius_max'] = np.max(all_temps)

      metrics['cpu_utilization_percent'] = psutil.cpu_percent(interval=0.1)
      metrics['cpu_utilization_percent_per_core'] = psutil.cpu_percent(interval=0.1, percpu=True)

      cpu_freq = psutil.cpu_freq()
      metrics['cpu_frequency_current_mhz'] = cpu_freq.current
      metrics['cpu_frequency_min_mhz'] = cpu_freq.min
      metrics['cpu_frequency_max_mhz'] = cpu_freq.max

      cpu_freq_core = psutil.cpu_freq(percpu=True)
      if cpu_freq_core:
        metrics['cpu_frequency_current_mhz_per_core'] = [f.current for f in cpu_freq_core]

      mem = psutil.virtual_memory()
      metrics['system_memory_total_mb'] = mem.total / (1024 * 1024)
      metrics['system_memory_used_mb'] = mem.used / (1024 * 1024)
      metrics['system_memory_free_mb'] = mem.free / (1024 * 1024)
      metrics['system_memory_utilization_percent'] = mem.percent

      try:
        swap = psutil.swap_memory()
        metrics['system_swap_total_mb'] = swap.total / (1024 * 1024)
        metrics['system_swap_used_mb'] = swap.used / (1024 * 1024)
        metrics['system_swap_percent'] = swap.percent
      except:
        pass

      battery = psutil.sensors_battery()
      if battery:
        metrics['battery_percent'] = battery.percent
        metrics['battery_plugged_in'] = battery.power_plugged
        metrics['battery_time_left_secs'] = battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None

      try:
        fans = psutil.sensors_fans()
        if fans:
          metrics['system_fan_speeds_rpm'] = {name: [fan.current for fan in fan_list] for name, fan_list in fans.items()}
      except:
        pass

      try:
        cpu_stats = psutil.cpu_stats()
        metrics['cpu_ctx_switches'] = cpu_stats.ctx_switches
        metrics['cpu_interrupts'] = cpu_stats.interrupts
        metrics['cpu_soft_interrupts'] = cpu_stats.soft_interrupts
        metrics['cpu_syscalls'] = cpu_stats.syscalls
      except:
        pass

      try:
        load_avg = psutil.getloadavg()
        metrics['system_load_1min'] = load_avg[0]
        metrics['system_load_5min'] = load_avg[1]
        metrics['system_load_15min'] = load_avg[2]
      except:
        pass

    except Exception as e:
      print(f"cpu system metrics collection failed: {e}")
      metrics['cpu_system_error'] = str(e)

    metrics['timestamp'] = datetime.now().isoformat()

    return metrics

  def get_normalized_state_vector(self):
    metrics = self.collect_metrics()
    state = []

    state.append(np.clip(metrics.get('gpu_temperature_celsius', 0) / self.baselines.get('gpu_temp_max', 85), 0, 1))
    state.append(metrics.get('gpu_utilization_percent', 0) / 100.0)
    state.append(metrics.get('gpu_memory_utilization_percent', 0) / 100.0)
    state.append(np.clip(metrics.get('gpu_power_watts', 0) / self.baselines.get('gpu_power_max', 300), 0, 1))
    state.append(metrics.get('gpu_fan_speed_percent', 0) / 100.0)
    state.append(metrics.get('gpu_sm_clock_mhz', 0) / self.baselines.get('gpu_sm_clock_max_mhz', 2000))
    state.append(metrics.get('gpu_memory_clock_mhz', 0) / self.baselines.get('gpu_memory_clock_max_mhz', 8000))

    state.append(np.clip(metrics.get('cpu_temperature_celsius_avg', 50) / self.baselines.get('cpu_temp_max', 95), 0, 1))
    state.append(metrics.get('cpu_utilization_percent', 0) / 100.0)
    state.append(metrics.get('cpu_frequency_current_mhz', 0) / self.baselines.get('cpu_frequency_max_mhz', 3000))

    state.append(metrics.get('system_memory_utilization_percent', 0) / 100.0)
    state.append(metrics.get('battery_percent', 100) / 100.0)
    state.append(1.0 if metrics.get('battery_plugged_in', True) else 0.0)

    pcie_bw_util = (metrics.get('pcie_tx_throughput_kbps', 0) + metrics.get('pcie_rx_throughput_kbps', 0)) / 32000000.0
    state.append(np.clip(pcie_bw_util, 0, 1))

    state.append(1.0 if metrics.get('gpu_throttling_active', False) else 0.0)

    return np.array(state, dtype=np.float32), metrics

  def compute_reward(self, energy_utilization, metrics):
    if 0.95 <= energy_utilization <= 0.98:
      reward = 1.0
    elif energy_utilization < 0.95:
      reward = energy_utilization / 0.95
    else:
      reward = max(0, 1.0 - (energy_utilization - 0.98) * 10)

    gpu_temp = metrics.get('gpu_temperature_celsius', 0)
    if gpu_temp > 80:
      reward *= 0.5
    elif gpu_temp > 75:
      reward *= 0.8

    if metrics.get('gpu_throttling_active', False):
      reward *= 0.3

    cpu_temp = metrics.get('cpu_temperature_celsius_avg', 0)
    if cpu_temp > 85:
      reward *= 0.6

    return np.clip(reward, 0, 1)


def profiler(func):
  profiler_instance = HardwareMetricsProfiler(gpu_index=0)

  @wraps(func)
  def wrapper(*args, **kwargs):
    print(f"\nprofiling {func.__name__}\n")

    metrics_before = profiler_instance.collect_metrics()

    energy_before = None
    if za:
      profiler_instance.zeus_monitor.begin_window(func.__name__)
    else:
      try:
        energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(profiler_instance.gpu_handle)
      except:
        pass

    start_time = datetime.now()
    result = func(*args, **kwargs)
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    energy_measurement = None
    if za:
      measurement = profiler_instance.zeus_monitor.end_window(func.__name__)
      energy_measurement = {
        'total_energy_joules': measurement.total_energy,
        'time_seconds': measurement.time,
        'average_power_watts': measurement.total_energy / measurement.time if measurement.time > 0 else 0,
      }
    elif energy_before is not None:
      try:
        energy_after = pynvml.nvmlDeviceGetTotalEnergyConsumption(profiler_instance.gpu_handle)
        energy_joules = (energy_after - energy_before) / 1000.0
        energy_measurement = {
          'total_energy_joules': energy_joules,
          'time_seconds': execution_time,
          'average_power_watts': energy_joules / execution_time if execution_time > 0 else 0,
        }
      except:
        pass

    metrics_after = profiler_instance.collect_metrics()

    deltas = {}
    for key in metrics_before:
      if key in metrics_after and isinstance(metrics_before[key], (int, float)):
        deltas[f"delta_{key}"] = metrics_after[key] - metrics_before[key]

    profile_data = {
      'function_name': func.__name__,
      'execution_time_seconds': execution_time,
      'timestamp_start': start_time.isoformat(),
      'timestamp_end': end_time.isoformat(),
      'energy_measurement': energy_measurement,
      'baselines': profiler_instance.baselines,
      'metrics_before': metrics_before,
      'metrics_after': metrics_after,
      'metrics_delta': deltas,
    }

    print("=" * 80)
    print(f"profile summary: {func.__name__}")
    print("=" * 80)
    print(f"execution time: {execution_time:.4f}s")

    if energy_measurement:
      print(f"\nenergy:")
      print(f"  total: {energy_measurement['total_energy_joules']:.2f}j")
      print(f"  avg power: {energy_measurement['average_power_watts']:.2f}w")

    print(f"\ngpu:")
    print(f"  temp: {metrics_before.get('gpu_temperature_celsius')}c -> {metrics_after.get('gpu_temperature_celsius')}c")
    print(f"  power: {metrics_before.get('gpu_power_watts'):.1f}w -> {metrics_after.get('gpu_power_watts'):.1f}w")
    print(f"  util: {metrics_before.get('gpu_utilization_percent')}% -> {metrics_after.get('gpu_utilization_percent')}%")
    print(f"  mem: {metrics_after.get('gpu_memory_used_mb'):.0f}mb / {metrics_after.get('gpu_memory_total_mb'):.0f}mb")
    print(f"  sm clock: {metrics_after.get('gpu_sm_clock_mhz')}mhz")
    print(f"  mem clock: {metrics_after.get('gpu_memory_clock_mhz')}mhz")
    print(f"  pcie: gen{metrics_after.get('pcie_current_gen')} x{metrics_after.get('pcie_current_width')}")
    print(f"  throttling: {metrics_after.get('gpu_throttling_active')}")

    print(f"\ncpu:")
    print(f"  temp: {metrics_before.get('cpu_temperature_celsius_avg', 'n/a')} -> {metrics_after.get('cpu_temperature_celsius_avg', 'n/a')}")
    print(f"  util: {metrics_before.get('cpu_utilization_percent'):.1f}% -> {metrics_after.get('cpu_utilization_percent'):.1f}%")
    print(f"  freq: {metrics_after.get('cpu_frequency_current_mhz'):.0f}mhz")

    if 'battery_percent' in metrics_after:
      print(f"\nbattery:")
      print(f"  charge: {metrics_after.get('battery_percent')}%")
      print(f"  plugged: {metrics_after.get('battery_plugged_in')}")

    print("=" * 80 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(profiler_instance.output_dir, f"{func.__name__}_profile_{timestamp}.json")

    with open(output_file, 'w') as f:
      json.dump(profile_data, f, indent=2, default=str)

    print(f"saved to {output_file}\n")

    return result

  return wrapper
