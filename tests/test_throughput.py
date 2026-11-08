"""Tests for the speed check."""

import pytest

from sizing import throughput


def test_time_per_token_worked_example():
    t = throughput.time_per_token_s(140, 268.435456, 8, 3350, 0.6)
    assert t == pytest.approx(0.0254, abs=0.00005)


def test_more_gpus_means_less_time():
    slow = throughput.time_per_token_s(140, 0, 4, 3350, 0.6)
    fast = throughput.time_per_token_s(140, 0, 8, 3350, 0.6)
    assert fast < slow


def test_time_per_token_zero_gpus_raises():
    with pytest.raises(ValueError):
        throughput.time_per_token_s(140, 0, 0, 3350, 0.6)


def test_tokens_per_s_per_user():
    assert throughput.tokens_per_s_per_user(0.0254) == pytest.approx(39.37, abs=0.01)


def test_tokens_per_s_zero_time_raises():
    with pytest.raises(ValueError):
        throughput.tokens_per_s_per_user(0)


def test_speed_check_passes():
    assert throughput.speed_check(39.4, 20) == {"value": 39.4, "target": 20, "passes": True}


def test_speed_check_fails_at_target_50():
    assert throughput.speed_check(39.4, 50)["passes"] is False


def test_speed_check_exactly_on_target_passes():
    assert throughput.speed_check(20, 20)["passes"] is True
