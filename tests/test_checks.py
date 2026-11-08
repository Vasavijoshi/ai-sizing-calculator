"""Tests for the plain-words warnings and the safe engine call."""

from conftest import FAKE_BIG_MODEL, worked_inputs
from sizing import models
from ui import checks


def has(messages, words):
    """Return True if any message contains these words."""
    return any(words in message for message in messages)


# ---- One test per warning ----

def test_speed_check_failed(run):
    assert has(checks.result_warnings(run(target_tokens_per_s=50)), "Too slow")


def test_model_spans_servers(run):
    result = run(model=FAKE_BIG_MODEL, model_name=FAKE_BIG_MODEL["name"])
    assert has(checks.result_warnings(result), "split across servers")


def test_kv_cache_bigger_than_weights(run):
    # Worked example: KV 268.4 GB is bigger than weights 140 GB.
    assert has(checks.result_warnings(run()), "KV cache")


def test_tokens_above_128000(run):
    assert has(checks.result_warnings(run(users=1, tokens_per_user=130_000)), "128,000")


def test_users_below_1_blocks():
    assert has(checks.input_problems(users=0, tokens_per_user=4_096), "Users")


def test_tokens_below_1_blocks():
    assert has(checks.input_problems(users=200, tokens_per_user=0), "Tokens")


def test_engine_error_becomes_friendly_message(gpus, server):
    a100 = models.find_by_name(gpus, "A100 80GB SXM")
    result, error = checks.safe_size(worked_inputs(gpu_name="A100 80GB SXM"), a100, server)
    assert result is None
    assert "A100 80GB SXM" in error and "bandwidth_gb_s" in error


def test_empty_custom_fields_block_with_friendly_message():
    custom = {"name": "Custom", "params_b": 13, "layers": None, "kv_heads": 8, "head_size": None}
    assert checks.custom_model_problems(custom) == [
        "Please fill in every Custom model field. Missing: Layers, Head size."]


def test_full_custom_model_has_no_problems():
    custom = {"name": "Custom", "params_b": 13, "layers": 40, "kv_heads": 8, "head_size": 128}
    assert checks.custom_model_problems(custom) == []


def test_presets_are_not_checked_as_custom():
    assert checks.custom_model_problems({"name": "7B class", "layers": None}) == []


# ---- Quiet cases ----

def test_small_workload_has_no_warnings(run):
    assert checks.result_warnings(run(users=1)) == []


def test_good_inputs_have_no_problems():
    assert checks.input_problems(users=200, tokens_per_user=4_096) == []


def test_safe_size_returns_result_when_ok(gpus, server):
    h100 = models.find_by_name(gpus, "H100 SXM")
    result, error = checks.safe_size(worked_inputs(), h100, server)
    assert error is None and result["gpus"]["count"] == 8
