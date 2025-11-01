import torch
from torch.profiler import profile, ProfilerActivity, record_function

from functools import wraps

def profiler(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    with profile(
      activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
      profile_memory = True,
      record_shapes = True,
      with_stack = True,
    ) as prof:
      with record_function(func.__name__):
        result = func(*args, **kwargs)

    print(f"Profling: {func.__name__}\n")
    print("\n")

    print("CUDA Table, first twenty values only")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print("\n")

    print("CPU Table, same as above bruh")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    print("\n")

    output_file = f"{func.__name__}_trace.json"
    prof.export_chrome_trace(output_file)

    return result

  return wrapper
