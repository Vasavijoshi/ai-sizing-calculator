"""Shared default settings. The UI can change any of these later with sliders."""

from sizing.models import require_fields

DEFAULTS = {
    "overhead": 1.15,               # extra 15% memory for runtime, activations, fragmentation
    "usable": 0.90,                 # only plan to fill 90% of each GPU's memory
    "bandwidth_efficiency": 0.6,    # real memory speed is about 60% of the spec-sheet peak
    "server_power_factor": 1.5,     # server power = GPU power x 1.5 (CPUs, fans, network, etc.)
    "traffic_bytes_per_token": 10,  # bytes sent to the user for each generated token
    "load_time_s": 120,             # seconds allowed to load the model weights from storage
    "model_versions": 3,            # how many copies/versions of the weights we keep on disk
    "log_extra": 1.2,               # extra 20% disk space for logs and other files
    "rack_kw": 15,                  # power one rack can supply and cool, in kW
    "target_tokens_per_s": 20,      # speed each user should get, in tokens per second
}


def settings(overrides=None):
    """Return the defaults with any overrides applied (unknown names raise ValueError)."""
    merged = dict(DEFAULTS)
    for name, value in (overrides or {}).items():
        if name not in DEFAULTS:
            raise ValueError(f"Unknown setting '{name}'. Use one of {list(DEFAULTS)}")
        merged[name] = value
    return merged


def gpu_spec(gpu, field):
    """Return one GPU field, or raise a clear error if it is still a TODO (null)."""
    require_fields(gpu, [field])
    return gpu[field]
