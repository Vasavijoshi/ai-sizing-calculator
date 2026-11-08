"""Plain-words warnings, input checks, and a safe engine call that never shows a stack trace."""

from sizing.engine import size_workload
from sizing.explain import num

LONG_CONTEXT_TOKENS = 128_000
CUSTOM_FIELD_LABELS = {"params_b": "Parameters", "layers": "Layers",
                       "kv_heads": "KV heads", "head_size": "Head size"}


def custom_model_problems(model):
    """Return a message if a Custom model still has empty fields (presets are skipped)."""
    if model.get("name") != "Custom":
        return []
    missing = [label for field, label in CUSTOM_FIELD_LABELS.items() if model.get(field) is None]
    if not missing:
        return []
    return [f"Please fill in every Custom model field. Missing: {', '.join(missing)}."]


def input_problems(users, tokens_per_user):
    """Return messages that block the calculation (empty list = OK to calculate)."""
    problems = []
    if users < 1:
        problems.append("Users must be at least 1. Please enter how many people will use the model.")
    if tokens_per_user < 1:
        problems.append("Tokens per user must be at least 1. Please enter the conversation length.")
    return problems


def speed_warning(result):
    """Return a warning if the speed is below the target, else None."""
    check = result["speed"]["check"]
    if check["passes"]:
        return None
    return (f"Too slow: each user gets about {num(check['value'])} tokens/s, below the target of "
            f"{num(check['target'])}. Try a GPU with faster memory, fewer users, shorter "
            "conversations, or a smaller number format.")


def spans_warning(result):
    """Return a warning if the model must be split across servers, else None."""
    if not result["gpus"]["spans_servers"]:
        return None
    return (f"The model needs {result['gpus']['raw']} GPUs, more than the "
            f"{result['server']['gpus_per_server']} in one server, so it is split across servers. "
            "Fast links between servers are needed.")


def kv_warning(result):
    """Return a note if the KV cache is bigger than the weights, else None."""
    m = result["memory"]
    if m["kv_gb"] <= m["weights_gb"]:
        return None
    return (f"The KV cache ({num(m['kv_gb'])} GB) is bigger than the model weights "
            f"({num(m['weights_gb'])} GB). Most memory goes to conversations, not the model.")


def context_warning(result):
    """Return a warning if tokens per user is above 128,000, else None."""
    tokens = result["inputs"]["tokens_per_user"]
    if tokens <= LONG_CONTEXT_TOKENS:
        return None
    return (f"{num(tokens, 0)} tokens per user is above {num(LONG_CONTEXT_TOKENS, 0)}. "
            "Check that the model supports such a long conversation.")


def result_warnings(result):
    """Return every warning that applies to this result, as a list of messages."""
    found = [speed_warning(result), spans_warning(result), kv_warning(result), context_warning(result)]
    return [message for message in found if message]


def safe_size(inputs, gpu, server, overrides=None, model=None):
    """Run the engine; return (result, None) or (None, a friendly error message)."""
    try:
        return size_workload(inputs, gpu, server, overrides=overrides, model=model), None
    except Exception as error:  # any engine error becomes a message, never a crash
        return None, f"We could not size this workload: {error}. Check the inputs or fill the missing value in the data files."
