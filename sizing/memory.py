"""GPU memory math. All results are indicative, in decimal GB (1 GB = 1e9 bytes)."""

BYTES_PER_GB = 1e9

BYTES_PER_PARAM = {"FP16": 2, "BF16": 2, "FP8": 1, "INT8": 1, "INT4": 0.5}


def bytes_per_param(precision):
    """Return how many bytes one number takes at this precision (e.g. FP16 -> 2)."""
    key = precision.upper()
    if key not in BYTES_PER_PARAM:
        raise ValueError(f"Unknown precision '{precision}'. Use one of {list(BYTES_PER_PARAM)}")
    return BYTES_PER_PARAM[key]


def check_not_negative(**values):
    """Raise ValueError if any given number is below zero."""
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative (got {value})")


def weights_gb(params_b, bytes_per_param):
    """Return the model weights size in GB (billions of params x bytes each)."""
    check_not_negative(params_b=params_b, bytes_per_param=bytes_per_param)
    return params_b * bytes_per_param


def kv_bytes_per_token(layers, kv_heads, head_size, bytes_per_value):
    """Return the KV cache bytes one token needs (2 = one K and one V)."""
    check_not_negative(layers=layers, kv_heads=kv_heads, head_size=head_size)
    return 2 * layers * kv_heads * head_size * bytes_per_value


def kv_total_gb(kv_per_token, tokens_per_user, users):
    """Return the KV cache size in GB for all users at full context."""
    check_not_negative(kv_per_token=kv_per_token, tokens_per_user=tokens_per_user, users=users)
    return kv_per_token * tokens_per_user * users / BYTES_PER_GB


def total_gpu_memory_gb(weights, kv_total, overhead=1.15):
    """Return weights plus KV cache, with extra room for overhead (default 15%)."""
    check_not_negative(weights=weights, kv_total=kv_total)
    return (weights + kv_total) * overhead
