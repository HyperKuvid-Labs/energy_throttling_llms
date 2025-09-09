import numpy as np

def collect_model_metrics(model, tokenizer, prompt, queue_length, rolling_latency):
    tokens = tokenizer.encode(prompt)
    metrics = {}
    metrics["prompt_length_tokens"] = len(tokens)

    # entropy / confidence requires log-probs
    with model.no_grad():
        outputs = model(tokens, labels=tokens)
        log_probs = outputs.logits.log_softmax(dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean().item()
    metrics["prompt_entropy"] = entropy

    metrics["model_id"] = model.config.model_type
    metrics["queue_length"] = queue_length
    metrics["rolling_avg_latency_ms"] = np.mean(rolling_latency)
    metrics["p95_latency_ms"] = np.percentile(rolling_latency, 95)

    return metrics
