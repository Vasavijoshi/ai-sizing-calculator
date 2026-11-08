"""Tests for the 'why this number' explanation steps."""

import pytest

from sizing import models
from sizing.engine import Inputs, size_workload
from sizing.explain import explain, num

H100 = models.find_by_name(models.load_gpus(), "H100 SXM")
XE9680 = models.find_by_name(models.load_servers(), "Dell PowerEdge XE9680")
KEYS = {"title", "formula", "filled_in", "result", "assumptions"}


@pytest.fixture(scope="module")
def steps():
    inputs = Inputs(model_name="Llama 3 70B class", users=200, tokens_per_user=4_096,
                    gpu_name="H100 SXM", backup_mode="n+1")
    return explain(size_workload(inputs, H100, XE9680))


def find_step(steps, title):
    """Return the step with this title."""
    return next(step for step in steps if step["title"] == title)


@pytest.mark.parametrize("value, digits, expected", [
    (140.0, 1, "140"), (268.435456, 1, "268.4"), (4096, 0, "4,096"),
    (0.025398, 4, "0.0254"), (1.15, 2, "1.15"), (100, 1, "100"),
])
def test_num_formats_for_people(value, digits, expected):
    assert num(value, digits) == expected


def test_every_step_has_all_five_keys(steps):
    for step in steps:
        assert set(step) == KEYS


def test_every_step_has_text_and_assumptions(steps):
    for step in steps:
        for key in ["title", "formula", "filled_in", "result"]:
            assert isinstance(step[key], str) and step[key]
        assert isinstance(step["assumptions"], list) and step["assumptions"]


def test_one_step_per_calculation(steps):
    titles = [step["title"] for step in steps]
    assert len(titles) == 15
    assert len(set(titles)) == 15


def test_filled_in_contains_the_real_input_numbers(steps):
    assert "70 billion" in find_step(steps, "Model weights")["filled_in"]
    kv_token = find_step(steps, "KV cache per token")["filled_in"]
    assert "80" in kv_token and "8" in kv_token and "128" in kv_token
    kv_total = find_step(steps, "KV cache for all users")["filled_in"]
    assert "4,096" in kv_total and "200 users" in kv_total
    assert "3,350" in find_step(steps, "Speed per user")["filled_in"]
    assert "700 W" in find_step(steps, "Power per server")["filled_in"]
    assert "15 kW" in find_step(steps, "Racks")["filled_in"]


def test_results_match_the_worked_example(steps):
    assert find_step(steps, "Model weights")["result"] == "140 GB"
    assert find_step(steps, "KV cache per token")["result"] == "327,680 bytes"
    assert find_step(steps, "Total GPU memory")["result"] == "469.7 GB"
    assert "8" in find_step(steps, "GPUs")["result"]
    assert find_step(steps, "Servers")["result"] == "2 servers"
    assert "passes" in find_step(steps, "Speed per user")["result"]


def test_assumptions_name_the_defaults_used(steps):
    assert "1.15" in " ".join(find_step(steps, "Total GPU memory")["assumptions"])
    assert "0.9" in " ".join(find_step(steps, "GPUs")["assumptions"])
    assert "0.6" in " ".join(find_step(steps, "Speed per user")["assumptions"])
