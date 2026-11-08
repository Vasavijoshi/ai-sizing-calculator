"""Tests for loading the YAML data files."""

import pytest

from sizing import models


def test_every_gpu_has_the_needed_fields():
    for gpu in models.load_gpus():
        for field in ["name", "memory_gb", "bandwidth_gb_s", "tdp_w", "source", "checked_on"]:
            assert field in gpu


def test_every_server_has_the_needed_fields():
    for server in models.load_servers():
        for field in ["name", "gpus_per_server", "form_factor_u", "cooling", "source", "checked_on"]:
            assert field in server


def test_every_model_has_the_needed_fields():
    for model in models.load_models():
        for field in ["name", "params_b", "layers", "kv_heads", "head_size", "source", "checked_on"]:
            assert field in model


def test_find_h100_by_name():
    gpu = models.find_by_name(models.load_gpus(), "H100 SXM")
    assert gpu["memory_gb"] == 80


def test_find_unknown_name_raises():
    with pytest.raises(KeyError):
        models.find_by_name(models.load_gpus(), "Not A GPU")


def test_require_fields_passes_for_70b():
    model = models.find_by_name(models.load_models(), "Llama 3 70B class")
    models.require_fields(model, ["layers", "kv_heads", "head_size"])


def test_require_fields_fails_for_todo_preset():
    model = models.find_by_name(models.load_models(), "7B class")
    with pytest.raises(ValueError, match="layers"):
        models.require_fields(model, ["layers", "kv_heads", "head_size"])
