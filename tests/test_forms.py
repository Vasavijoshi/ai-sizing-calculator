"""Tests for the form helpers (no Streamlit needed)."""

from sizing.defaults import DEFAULTS
from sizing.engine import Inputs
from ui import forms

# Small made-up rows, so these tests keep passing when you fill the real YAML TODOs.
FAKE_MODELS = [
    {"name": "Full model", "params_b": 10, "layers": 40, "kv_heads": 8, "head_size": 128},
    {"name": "Half model", "params_b": 20, "layers": None, "kv_heads": None, "head_size": None},
]
FAKE_GPUS = [
    {"name": "Full GPU", "memory_gb": 80, "bandwidth_gb_s": 1000, "tdp_w": 500},
    {"name": "No speed GPU", "memory_gb": 80, "bandwidth_gb_s": None, "tdp_w": 500},
    {"name": "No power GPU", "memory_gb": 80, "bandwidth_gb_s": 1000, "tdp_w": None},
]


def test_model_choices_are_complete_presets_plus_custom():
    assert forms.model_choices(FAKE_MODELS) == ["Full model", "Custom"]


def test_incomplete_models_are_listed():
    assert forms.incomplete_models(FAKE_MODELS) == ["Half model"]


def test_gpu_choices_need_bandwidth_and_power():
    assert forms.gpu_choices(FAKE_GPUS) == ["Full GPU"]
    assert forms.incomplete_gpus(FAKE_GPUS) == ["No speed GPU", "No power GPU"]


def test_needs_config_caption():
    assert forms.needs_config_caption(["A", "B"]) == "needs config: A, B"


def test_needs_config_caption_empty():
    assert forms.needs_config_caption([]) == ""


def test_precision_labels_map_to_engine_names():
    assert forms.PRECISION_OPTIONS == {"FP16/BF16": "FP16", "FP8/INT8": "FP8", "INT4": "INT4"}


def test_custom_model_row():
    row = forms.custom_model(params_b=13, layers=40, kv_heads=8, head_size=128)
    assert row == {"name": "Custom", "params_b": 13, "layers": 40, "kv_heads": 8, "head_size": 128}


def test_pick_model_named():
    assert forms.pick_model(FAKE_MODELS, "Full model")["params_b"] == 10


def test_pick_model_custom():
    custom = forms.custom_model(13, 40, 8, 128)
    assert forms.pick_model(FAKE_MODELS, "Custom", custom) is custom


def test_slider_start_values_come_from_defaults():
    for name, slider in forms.SLIDERS.items():
        assert forms.slider_start(name) == DEFAULTS[name]
        assert slider["min"] <= DEFAULTS[name] <= slider["max"]


def test_build_overrides_keeps_only_slider_settings():
    values = {"overhead": 1.2, "usable": 0.8, "bandwidth_efficiency": 0.5,
              "server_power_factor": 1.6, "something_else": 9}
    assert forms.build_overrides(values) == {"overhead": 1.2, "usable": 0.8,
                                             "bandwidth_efficiency": 0.5, "server_power_factor": 1.6}


def test_build_inputs():
    inputs = forms.build_inputs(model_name="Full model", precision_label="FP8/INT8", users=50,
                                tokens_per_user=2048, target=25, gpu_name="Full GPU",
                                backup_mode="n+1", rack_kw=20, load_time_s=60)
    assert inputs == Inputs(model_name="Full model", users=50, tokens_per_user=2048, precision="FP8",
                            target_tokens_per_s=25, gpu_name="Full GPU", backup_mode="n+1",
                            rack_kw=20, load_time_s=60)
