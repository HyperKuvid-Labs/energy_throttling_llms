from RL.components.profiler_cpu_gpu import profiler

@profiler
def test_function(x):
  import torch
  x = torch.randn((8192, 8192), device="cuda")
  for i in range(100):
    x = torch.matmul(x, x)
  return x

test_function(10)