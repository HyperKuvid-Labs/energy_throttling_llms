def collect_eagle_metrics(stats):
    """
    stats: dict provided by SGLang speculative decoding hooks
    Expected keys: branches, accepted, cascades
    """
    metrics = {}
    metrics["spec_branches"] = stats.get("branches", 0)
    metrics["spec_accept_rate_pct"] = (stats.get("accepted", 0) / max(1, stats.get("attempted", 1))) * 100
    metrics["rejection_cascade_depth"] = stats.get("cascades", 0)
    return metrics
