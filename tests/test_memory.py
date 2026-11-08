"""Tests for the GPU memory math."""

import pytest

from sizing import memory


@pytest.mark.parametrize("precision, expected", [
    ("FP16", 2), ("BF16", 2), ("FP8", 1), ("INT8", 1), ("INT4", 0.5), ("fp16", 2),
])
def test_bytes_per_param(precision, expected):
    assert memory.bytes_per_param(precision) == expected


def test_bytes_per_param_unknown_raises():
    with pytest.raises(ValueError):
        memory.bytes_per_param("FP3")


def test_weights_gb_70b_fp16():
    assert memory.weights_gb(70, 2) == 140


def test_weights_gb_70b_int4():
    assert memory.weights_gb(70, 0.5) == 35


def test_kv_bytes_per_token_70b():
    assert memory.kv_bytes_per_token(80, 8, 128, 2) == 327_680


def test_kv_total_gb_converts_to_decimal_gb():
    # 1,000,000 bytes per token x 1,000 tokens x 1 user = 1e9 bytes = 1 GB
    assert memory.kv_total_gb(1_000_000, 1_000, 1) == 1


def test_kv_total_gb_zero_users_is_zero():
    assert memory.kv_total_gb(327_680, 4_096, 0) == 0


def test_total_gpu_memory_adds_15_percent():
    assert memory.total_gpu_memory_gb(100, 100) == pytest.approx(230)


def test_total_gpu_memory_custom_overhead():
    assert memory.total_gpu_memory_gb(100, 0, overhead=1.0) == 100


def test_negative_inputs_raise():
    with pytest.raises(ValueError):
        memory.kv_total_gb(327_680, 4_096, -1)
