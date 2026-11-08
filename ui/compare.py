"""Compare GPUs: run the same workload on 2 or 3 GPUs and build one table."""

from dataclasses import replace

import pandas as pd

from ui.checks import safe_size
from ui.display import speed_status

COLUMNS = ["GPU", "GPUs", "Servers", "Speed (tokens/s per user)", "Speed check",
           "Total power (kW)", "Racks"]


def choice_problem(names):
    """Return a message if not 2 or 3 GPUs are picked, else None."""
    return None if 2 <= len(names) <= 3 else "Pick 2 or 3 GPUs to compare."


def result_row(result):
    """Return one table row (dict) from an engine result."""
    return {"GPU": result["gpu"]["name"], "GPUs": result["gpus"]["count"],
            "Servers": result["servers"]["after_backup"],
            "Speed (tokens/s per user)": round(result["speed"]["tokens_per_s_per_user"], 1),
            "Speed check": speed_status(result), "Total power (kW)": round(result["power"]["total_kw"], 1),
            "Racks": result["power"]["racks"]}


def error_row(gpu_name, message):
    """Return a table row for a GPU that could not be sized, with the reason."""
    return {"GPU": gpu_name, "GPUs": None, "Servers": None, "Speed (tokens/s per user)": None,
            "Speed check": message, "Total power (kW)": None, "Racks": None}


def compare_table(inputs, gpus, server, overrides=None, model=None):
    """Return a DataFrame with one row per GPU, same workload for all."""
    rows = []
    for gpu in gpus:
        result, error = safe_size(replace(inputs, gpu_name=gpu["name"]), gpu, server, overrides, model)
        rows.append(error_row(gpu["name"], error) if error else result_row(result))
    return pd.DataFrame(rows, columns=COLUMNS)
