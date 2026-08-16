"""Reward for a single sweep row.

The README's target -- "keep energy utilization in a tight 95-98% band" -- was
never defined anywhere in the codebase. Here it means:

    energy_utilization = avg_power_draw_during_inference / gpu_power_limit

both in watts, with the limit read from NVML
(nvmlDeviceGetPowerManagementLimit; 400 W on the A100-SXM4-80GB used for
collection). This is the only reading that survives the move to datacenter
hardware: there is no battery to drain, and A100 SXM is passively cooled so fan
speed reads as unsupported.

Rationale for the band: below it the GPU is idle capacity the policy should
spend on deeper speculation; above it the card approaches its power cap and
clocks down, so more speculation buys nothing and costs energy.

Throughput is what stops the policy from satisfying the band by simply doing
useless work, so the two terms are multiplied rather than added -- a config that
sits perfectly in the band but decodes slower than the non-speculative baseline
must not score well. Thermal and throttle terms are multiplicative penalties
carried over from HardwareMetricsProfiler.compute_reward, which was never wired
to anything.
"""

import numpy as np

ENERGY_BAND = (0.95, 0.98)
OVERSHOOT_SLOPE = 10.0   # reward hits 0 at 10.8% over the band
GPU_TEMP_WARN = 75.0
GPU_TEMP_HOT = 80.0


def energy_utilization(avg_power_watts, power_limit_watts):
    if not power_limit_watts:
        return None
    return avg_power_watts / power_limit_watts


def band_score(utilization):
    """1.0 inside the band, ramping in from below, falling off sharply above."""
    lo, hi = ENERGY_BAND
    if utilization is None:
        return 0.0
    if lo <= utilization <= hi:
        return 1.0
    if utilization < lo:
        return utilization / lo
    return max(0.0, 1.0 - (utilization - hi) * OVERSHOOT_SLOPE)


def throughput_score(speed_tok_s, baseline_speed_tok_s):
    """Speedup over the non-speculative baseline, saturating at 3x.

    Clipped rather than unbounded so a single fast config cannot dominate the
    objective, and floored at 0 so configs slower than baseline score zero
    instead of going negative and fighting the band term.
    """
    if not speed_tok_s or not baseline_speed_tok_s:
        return 0.0
    return float(np.clip((speed_tok_s / baseline_speed_tok_s - 1.0) / 2.0, 0.0, 1.0))


def thermal_multiplier(gpu_temp_c, throttling):
    m = 1.0
    if gpu_temp_c is not None:
        if gpu_temp_c > GPU_TEMP_HOT:
            m *= 0.5
        elif gpu_temp_c > GPU_TEMP_WARN:
            m *= 0.8
    if throttling:
        m *= 0.3
    return m


def compute_reward(row, baseline_speed_tok_s, power_limit_watts, energy_weight=0.5):
    """Scalar reward in [0, 1] for one row of the sweep JSONL.

    energy_weight trades the band against raw speed; 0.5 weights them equally.
    Returns (reward, parts) so the parts can be inspected when tuning.
    """
    if row.get("error"):
        return 0.0, {"reason": "errored config"}

    util = energy_utilization(row.get("avg_power_watts"), power_limit_watts)
    e = band_score(util)
    t = throughput_score(row.get("speed_tok_s"), baseline_speed_tok_s)

    # Geometric mean: both terms must be decent. An arithmetic mean would let a
    # config that ignores one objective entirely still score 0.5.
    combined = (e ** energy_weight) * (t ** (1.0 - energy_weight))
    reward = combined * thermal_multiplier(row.get("gpu_temp_c"), row.get("gpu_throttling"))

    return float(np.clip(reward, 0.0, 1.0)), {
        "energy_utilization": util,
        "band_score": e,
        "throughput_score": t,
        "thermal_multiplier": thermal_multiplier(row.get("gpu_temp_c"), row.get("gpu_throttling")),
    }
