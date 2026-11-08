"""Helpers for the sidebar form: what can be picked, and turning answers into Inputs."""

from sizing.defaults import DEFAULTS
from sizing.engine import MODEL_FIELDS, Inputs
from sizing.models import find_by_name

CUSTOM = "Custom"
GPU_FIELDS = ["memory_gb", "bandwidth_gb_s", "tdp_w"]
PRECISION_OPTIONS = {"FP16/BF16": "FP16", "FP8/INT8": "FP8", "INT4": "INT4"}
BACKUP_OPTIONS = ["none", "n+1", "double"]

# Starting values for the workload (these are not assumptions, so not in defaults.py).
START_USERS = 200
START_TOKENS = 4_096
USERS_MIN, USERS_MAX = 1, 5_000

# Slider ranges for the "Advanced assumptions" expander. Start values come from DEFAULTS.
SLIDERS = {
    "overhead": {"label": "Extra memory (overhead)", "min": 1.05, "max": 1.5, "step": 0.01},
    "usable": {"label": "Usable share of GPU memory", "min": 0.70, "max": 0.95, "step": 0.01},
    "bandwidth_efficiency": {"label": "Speed efficiency", "min": 0.3, "max": 0.9, "step": 0.05},
    "server_power_factor": {"label": "Server power factor", "min": 1.2, "max": 2.0, "step": 0.05},
}


def is_complete(row, fields):
    """Return True if none of these fields is null."""
    return all(row.get(field) is not None for field in fields)


def model_choices(models):
    """Return the names of complete model presets, plus 'Custom'."""
    return [m["name"] for m in models if is_complete(m, MODEL_FIELDS)] + [CUSTOM]


def incomplete_models(models):
    """Return the names of model presets that still have null fields."""
    return [m["name"] for m in models if not is_complete(m, MODEL_FIELDS)]


def gpu_choices(gpus):
    """Return the names of GPUs with memory, bandwidth and power all filled in."""
    return [g["name"] for g in gpus if is_complete(g, GPU_FIELDS)]


def incomplete_gpus(gpus):
    """Return the names of GPUs that still have null fields."""
    return [g["name"] for g in gpus if not is_complete(g, GPU_FIELDS)]


def needs_config_caption(names):
    """Return 'needs config: a, b' text, or an empty string if nothing is missing."""
    return f"needs config: {', '.join(names)}" if names else ""


def custom_model(params_b, layers, kv_heads, head_size):
    """Return a model row built from the Custom fields."""
    return {"name": CUSTOM, "params_b": params_b, "layers": layers,
            "kv_heads": kv_heads, "head_size": head_size}


def pick_model(models, choice, custom=None):
    """Return the chosen model row (the custom row if 'Custom' was chosen)."""
    return custom if choice == CUSTOM else find_by_name(models, choice)


def slider_start(name):
    """Return a slider's starting value, taken from the shared defaults."""
    return DEFAULTS[name]


def build_overrides(values):
    """Return only the slider settings, ready to pass to size_workload."""
    return {name: values[name] for name in SLIDERS if name in values}


def build_inputs(model_name, precision_label, users, tokens_per_user, target,
                 gpu_name, backup_mode, rack_kw, load_time_s):
    """Turn the form answers into an engine Inputs object."""
    return Inputs(model_name=model_name, users=users, tokens_per_user=tokens_per_user,
                  precision=PRECISION_OPTIONS[precision_label], target_tokens_per_s=target,
                  gpu_name=gpu_name, backup_mode=backup_mode, rack_kw=rack_kw,
                  load_time_s=load_time_s)
