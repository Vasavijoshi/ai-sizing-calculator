"""Speed check: how many tokens per second each user gets. Results are indicative."""


def time_per_token_s(weights_gb, kv_read_gb, gpus, bandwidth_gb_s, efficiency):
    """Return seconds per token: data read per token / real memory speed of all GPUs."""
    if gpus <= 0 or bandwidth_gb_s <= 0 or efficiency <= 0:
        raise ValueError("gpus, bandwidth_gb_s and efficiency must all be above zero")
    return (weights_gb + kv_read_gb) / (gpus * bandwidth_gb_s * efficiency)


def tokens_per_s_per_user(time_per_token):
    """Return tokens per second each user gets (1 / seconds per token)."""
    if time_per_token <= 0:
        raise ValueError("time_per_token must be above zero")
    return 1 / time_per_token


def speed_check(tokens_per_s, target):
    """Return the speed, the target, and whether the speed meets the target."""
    return {"value": tokens_per_s, "target": target, "passes": tokens_per_s >= target}
