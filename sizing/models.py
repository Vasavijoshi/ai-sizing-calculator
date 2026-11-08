"""Load GPU, server and model data from the YAML files in data/."""

from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_yaml(filename):
    """Read one YAML file from the data folder and return its list of rows."""
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gpus():
    """Return the list of GPUs."""
    return load_yaml("gpus.yaml")


def load_servers():
    """Return the list of servers."""
    return load_yaml("servers.yaml")


def load_models():
    """Return the list of model presets."""
    return load_yaml("models.yaml")


def find_by_name(rows, name):
    """Return the row whose name matches, or raise KeyError."""
    for row in rows:
        if row["name"] == name:
            return row
    raise KeyError(f"No entry named '{name}'")


def require_fields(row, fields):
    """Raise ValueError if any of these fields is still null (a TODO)."""
    missing = [field for field in fields if row.get(field) is None]
    if missing:
        raise ValueError(f"'{row['name']}' still has TODO fields: {', '.join(missing)}")
