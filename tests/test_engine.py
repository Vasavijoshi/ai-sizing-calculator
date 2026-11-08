"""Tests for size_workload: the whole engine in one call."""

import pytest

from sizing import models, storage_network
from sizing.engine import Inputs, size_workload

GPUS = models.load_gpus()
H100 = models.find_by_name(GPUS, "H100 SXM")
H200 = models.find_by_name(GPUS, "H200 SXM")
A100 = models.find_by_name(GPUS, "A100 80GB SXM")
XE9680 = models.find_by_name(models.load_servers(), "Dell PowerEdge XE9680")
LLAMA_70B = models.find_by_name(models.load_models(), "Llama 3 70B class")

# Not a real model: a made-up big model that is only used to test the "spans servers" rule.
FAKE_BIG_MODEL = {"name": "Fake big model (test only)", "params_b": 400,
                  "layers": 120, "kv_heads": 8, "head_size": 128}


def worked_inputs(**changes):
    """Return the worked-example inputs, with any changes applied."""
    values = dict(model_name="Llama 3 70B class", users=200, tokens_per_user=4_096,
                  precision="FP16", target_tokens_per_s=20, gpu_name="H100 SXM",
                  backup_mode="none")
    values.update(changes)
    return Inputs(**values)


@pytest.fixture(scope="module")
def worked():
    return size_workload(worked_inputs(), H100, XE9680)


@pytest.fixture(scope="module")
def worked_n_plus_1():
    return size_workload(worked_inputs(backup_mode="n+1"), H100, XE9680)


# ---- Worked example, backup "none" ----

def test_memory_breakdown(worked):
    m = worked["memory"]
    assert m["weights_gb"] == 140
    assert m["kv_gb"] == pytest.approx(268.4, abs=0.05)
    assert m["extra_gb"] == pytest.approx(469.7 - 408.4, abs=0.1)
    assert m["total_gb"] == pytest.approx(469.7, abs=0.05)


def test_gpus_and_servers(worked):
    assert worked["gpus"]["count"] == 8
    assert worked["servers"]["before_backup"] == 1
    assert worked["servers"]["after_backup"] == 1


def test_speed(worked):
    s = worked["speed"]
    assert s["time_per_token_s"] == pytest.approx(0.0254, abs=0.00005)
    assert s["tokens_per_s_per_user"] == pytest.approx(39.4, abs=0.05)
    assert s["check"]["passes"] is True


def test_storage(worked):
    assert worked["storage"]["read_gb_s"] == pytest.approx(1.17, abs=0.005)
    assert worked["storage"]["size_gb"] == pytest.approx(504)


def test_network(worked):
    assert worked["network"]["user_traffic_mbps"] == pytest.approx(0.63, abs=0.005)
    assert worked["network"]["advice"] == storage_network.ADVICE_ONE_SERVER


def test_power(worked):
    p = worked["power"]
    assert p["per_server_kw"] == pytest.approx(8.4)
    assert p["btu_per_hr"] == pytest.approx(28_661, abs=1)
    assert p["racks"] == 1


def test_result_is_marked_indicative(worked):
    assert worked["indicative"] is True


# ---- Same case with backup "n+1" ----

def test_n_plus_1(worked_n_plus_1):
    r = worked_n_plus_1
    assert r["servers"]["after_backup"] == 2
    assert r["power"]["total_kw"] == pytest.approx(16.8)
    assert r["storage"]["read_gb_s"] == pytest.approx(2.33, abs=0.005)
    assert r["power"]["racks"] == 2


# ---- Same case on H200 ----

def test_h200():
    r = size_workload(worked_inputs(gpu_name="H200 SXM"), H200, XE9680)
    assert r["gpus"]["count"] == 4
    assert r["speed"]["tokens_per_s_per_user"] == pytest.approx(28.2, abs=0.05)
    assert r["speed"]["check"]["passes"] is True


# ---- Extra tests ----

def test_a100_null_bandwidth_raises_clear_error():
    with pytest.raises(ValueError, match="A100 80GB SXM.*bandwidth_gb_s"):
        size_workload(worked_inputs(gpu_name="A100 80GB SXM"), A100, XE9680)


def test_speed_check_fails_at_target_50():
    r = size_workload(worked_inputs(target_tokens_per_s=50), H100, XE9680)
    assert r["speed"]["check"]["passes"] is False


def test_total_power_never_goes_down_as_users_grow():
    previous = 0
    for users in range(1, 2001):
        r = size_workload(worked_inputs(users=users), H100, XE9680, model=LLAMA_70B)
        assert r["power"]["total_kw"] >= previous, f"power dropped at {users} users"
        previous = r["power"]["total_kw"]


def test_big_model_spans_servers():
    inputs = worked_inputs(model_name=FAKE_BIG_MODEL["name"])
    r = size_workload(inputs, H100, XE9680, model=FAKE_BIG_MODEL)
    assert r["gpus"]["raw"] > 8
    assert r["gpus"]["spans_servers"] is True
    assert r["network"]["advice"] == storage_network.ADVICE_SPANS_SERVERS


def test_overrides_change_the_result():
    r = size_workload(worked_inputs(), H100, XE9680, overrides={"overhead": 1.0})
    assert r["memory"]["total_gb"] == pytest.approx(408.4, abs=0.05)


def test_gpu_row_must_match_inputs_gpu_name():
    with pytest.raises(ValueError, match="H200 SXM"):
        size_workload(worked_inputs(gpu_name="H200 SXM"), H100, XE9680)


def test_todo_model_preset_raises_clear_error():
    with pytest.raises(ValueError, match="7B class.*layers"):
        size_workload(worked_inputs(model_name="7B class"), H100, XE9680)
