"""The worked example: Llama 3 70B class, FP16, 4,096 tokens, 200 users, H100 80 GB."""

import pytest

from sizing import cluster, memory, models

MODEL = models.find_by_name(models.load_models(), "Llama 3 70B class")
GPU = models.find_by_name(models.load_gpus(), "H100 SXM")
FP16 = memory.bytes_per_param("FP16")
KV_PER_TOKEN = memory.kv_bytes_per_token(MODEL["layers"], MODEL["kv_heads"], MODEL["head_size"], FP16)


def total_gb_for(users, tokens_per_user=4_096):
    """Return the total GPU memory (GB) for the 70B model at FP16."""
    weights = memory.weights_gb(MODEL["params_b"], FP16)
    kv_total = memory.kv_total_gb(KV_PER_TOKEN, tokens_per_user, users)
    return memory.total_gpu_memory_gb(weights, kv_total)


def gpus_for(users, tokens_per_user=4_096):
    """Return the rounded GPU count on H100 for this many users."""
    return cluster.gpus_needed(total_gb_for(users, tokens_per_user), GPU["memory_gb"])


# ---- The worked example, step by step ----

def test_weights_are_140_gb():
    assert memory.weights_gb(MODEL["params_b"], FP16) == 140


def test_kv_per_token_is_327680_bytes():
    assert KV_PER_TOKEN == 327_680


def test_kv_total_is_about_268_4_gb():
    assert memory.kv_total_gb(KV_PER_TOKEN, 4_096, 200) == pytest.approx(268.4, abs=0.05)


def test_total_with_15_percent_is_about_469_7_gb():
    assert total_gb_for(200) == pytest.approx(469.7, abs=0.05)


def test_usable_per_gpu_is_72_gb():
    assert GPU["memory_gb"] * 0.90 == pytest.approx(72)


def test_raw_gpu_need_is_about_6_5():
    assert total_gb_for(200) / 72 == pytest.approx(6.5, abs=0.05)


def test_rounded_result_is_8_gpus():
    assert gpus_for(200) == 8


def test_servers_is_1_and_2_with_n_plus_1():
    servers = cluster.servers_needed(gpus_for(200))
    assert servers == 1
    assert cluster.apply_backup(servers, "n+1") == 2


# ---- Direction test: more users never means fewer GPUs ----

def test_gpu_count_never_goes_down_as_users_grow():
    previous = 0
    for users in range(1, 2001):
        current = gpus_for(users)
        assert current >= previous, f"GPU count dropped at {users} users"
        previous = current


# ---- Edge cases ----

def test_zero_users_still_needs_room_for_weights():
    # 140 x 1.15 = 161 GB -> 2.24 GPUs -> 3 -> rounded to 4
    assert total_gb_for(0) == pytest.approx(161)
    assert gpus_for(0) == 4


def test_one_user():
    # 140 + 1.34 GB KV, x 1.15 = about 162.5 GB -> 4 GPUs
    assert total_gb_for(1) == pytest.approx(162.5, abs=0.05)
    assert gpus_for(1) == 4


def test_one_user_with_128000_tokens():
    # KV = 327,680 x 128,000 = about 41.9 GB; total about 209.2 GB -> 4 GPUs
    assert memory.kv_total_gb(KV_PER_TOKEN, 128_000, 1) == pytest.approx(41.9, abs=0.05)
    assert total_gb_for(1, tokens_per_user=128_000) == pytest.approx(209.2, abs=0.05)
    assert gpus_for(1, tokens_per_user=128_000) == 4
