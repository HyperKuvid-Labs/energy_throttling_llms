import psutil
import time

def collect_cpu_metrics():
    metrics = {}
    metrics["cpu_utilization_pct"] = psutil.cpu_percent(interval=0.1)
    metrics["process_cpu_time_s"] = psutil.Process().cpu_times().user
    metrics["system_ram_used_mib"] = psutil.virtual_memory().used / (1024**2)
    metrics["swap_usage_mib"] = psutil.swap_memory().used / (1024**2)
    metrics["page_faults"] = psutil.swap_memory().sin
    metrics["disk_io_mb_s"] = psutil.disk_io_counters().read_bytes / (1024**2)
    metrics["net_io_mb_s"] = psutil.net_io_counters().bytes_sent / (1024**2)
    return metrics
