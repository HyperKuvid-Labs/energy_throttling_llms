# Parameters

Lets see what and all parameters we get as a verfied reward to train the RL excluding maybe the output from the LM head inside that sg lang thing.

## So first GPU level:
1. GPU power draw (W)
2. GPU utilization (%) 
3. GPU memory used (MiB) 
4. GPU memory utilization (%) 
5. GPU temperature (°C) 
6. GPU clock / mem clock (MHz) 
7. Fan speed (%) 
8. PCIe throughput (MB/s)
9. CUDA kernel time per batch (ms) 
10. CUDA kernel launch latency (ms) 
11. Number of CUDA streams active 
12. GPU SM occupancy / efficiency

## CPU level:
1. CPU utilization (%) 
2. Process CPU time (s)
3. System RAM used (MiB) 
4. Swap usage (MiB) / page faults 
5. Disk I/O (MB/s)
6. Network I/O (MB/s)

## Infernce runtime based on toknes:
1. Token generation throughput (tokens/sec) -> calculate using (num_tokens / elapsed_time)
2. Time to first token (ms) -> capture timestamp from request start to the first output 
3. Mean inter-token time (ms)
4. Total inference time (ms) for request 
5. Energy per token (J/token) -> calculate using (avg power * elapsed_time) / num_tokens.

## EAGLE level for decoding-specific ... we can use SGLang internal hooks:
1. Number of speculative branches created (count) 
2. Speculative acceptance rate (%) 
3) Rejection cascade depth / count

## Model based parameters I can think of :
1. Input prompt length (tokens)
2. Estimated prompt perplexity / LM-confidence / entropy ->how surprising the input prompt is to the model.
3. Model currently used (one-hot/embedding)
4. Current queue length / concurrent requests 
5. Historical rolling average latency / p95 latency