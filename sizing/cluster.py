"""Turn a memory need (GB) into GPU and server counts. Results are indicative."""

import math

GPU_STEPS = [1, 2, 4, 8]


def round_up_gpu_count(raw):
    """Round up to 1, 2, 4 or 8 GPUs; above 8, round up to a multiple of 8."""
    if raw <= 0:
        return 0
    for step in GPU_STEPS:
        if raw <= step:
            return step
    return math.ceil(raw / 8) * 8


def raw_gpus(total_gb, gpu_memory_gb, usable=0.90):
    """Return GPUs needed before rounding: ceil(total / usable memory per GPU)."""
    if not 0 < usable <= 1:
        raise ValueError(f"usable must be between 0 and 1 (got {usable})")
    return math.ceil(total_gb / (gpu_memory_gb * usable))


def gpus_needed(total_gb, gpu_memory_gb, usable=0.90):
    """Return how many GPUs hold total_gb, using only the usable share of each GPU."""
    return round_up_gpu_count(raw_gpus(total_gb, gpu_memory_gb, usable))


def servers_needed(gpus, gpus_per_server=8):
    """Return how many servers hold this many GPUs (always rounds up)."""
    return math.ceil(gpus / gpus_per_server)


def apply_backup(servers, mode):
    """Add spare servers: 'none' keeps it, 'n+1' adds one, 'double' doubles it."""
    mode = mode.lower()
    if mode == "none":
        return servers
    if mode == "n+1":
        return servers + 1
    if mode == "double":
        return servers * 2
    raise ValueError(f"Unknown backup mode '{mode}'. Use 'none', 'n+1' or 'double'")
