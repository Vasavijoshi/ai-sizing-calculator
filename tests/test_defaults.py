"""Tests for the shared default settings."""

import pytest

from sizing import defaults, models

GPUS = models.load_gpus()


def test_all_ten_defaults_are_there():
    expected = {
        "overhead": 1.15, "usable": 0.90, "bandwidth_efficiency": 0.6,
        "server_power_factor": 1.5, "traffic_bytes_per_token": 10, "load_time_s": 120,
        "model_versions": 3, "log_extra": 1.2, "rack_kw": 15, "target_tokens_per_s": 20,
    }
    assert defaults.DEFAULTS == expected


def test_settings_without_overrides_is_a_copy():
    s = defaults.settings()
    s["overhead"] = 9
    assert defaults.DEFAULTS["overhead"] == 1.15


def test_settings_override_changes_only_that_value():
    s = defaults.settings({"overhead": 1.2})
    assert s["overhead"] == 1.2
    assert s["usable"] == 0.90


def test_settings_unknown_key_raises():
    with pytest.raises(ValueError, match="overhed"):
        defaults.settings({"overhed": 1.2})


def test_gpu_spec_reads_a_filled_field():
    h100 = models.find_by_name(GPUS, "H100 SXM")
    assert defaults.gpu_spec(h100, "bandwidth_gb_s") == 3350


def test_gpu_spec_null_field_names_gpu_and_field():
    a100 = models.find_by_name(GPUS, "A100 80GB SXM")
    with pytest.raises(ValueError, match="A100 80GB SXM.*bandwidth_gb_s"):
        defaults.gpu_spec(a100, "bandwidth_gb_s")
