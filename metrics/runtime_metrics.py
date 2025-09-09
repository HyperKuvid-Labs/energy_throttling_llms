import time

class RuntimeTracker:
    def __init__(self, num_tokens: int):
        self.start_time = time.time()
        self.first_token_time = None
        self.last_token_time = None
        self.num_tokens = num_tokens

    def mark_first_token(self):
        self.first_token_time = time.time()

    def mark_token(self):
        self.last_token_time = time.time()

    def finalize(self, avg_power_w: float):
        end_time = time.time()
        elapsed = end_time - self.start_time

        metrics = {}
        metrics["throughput_tokens_sec"] = self.num_tokens / elapsed if elapsed > 0 else 0
        metrics["time_to_first_token_ms"] = (self.first_token_time - self.start_time) * 1000 if self.first_token_time else None
        metrics["mean_inter_token_time_ms"] = ((self.last_token_time - self.first_token_time) / max(1, self.num_tokens - 1)) * 1000 if self.first_token_time and self.last_token_time else None
        metrics["total_inference_time_ms"] = elapsed * 1000
        metrics["energy_per_token_j"] = (avg_power_w * elapsed) / self.num_tokens if self.num_tokens > 0 else None

        return metrics
