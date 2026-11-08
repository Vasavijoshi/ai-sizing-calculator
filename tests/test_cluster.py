"""Tests for turning GB into GPUs and servers."""

import pytest

from sizing import cluster


@pytest.mark.parametrize("raw, expected", [
    (0, 0), (1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (7, 8), (8, 8),
    (9, 16), (16, 16), (17, 24), (25, 32),
])
def test_round_up_gpu_count(raw, expected):
    assert cluster.round_up_gpu_count(raw) == expected


def test_raw_gpus_before_rounding():
    # 469.7 GB / 72 GB usable = 6.5 -> 7 (before rounding to 8)
    assert cluster.raw_gpus(469.7, 80) == 7


def test_gpus_needed_fits_on_one_gpu():
    assert cluster.gpus_needed(50, 80) == 1


def test_gpus_needed_uses_90_percent_of_memory():
    # 75 GB is less than 80 GB, but more than the usable 72 GB -> 2 GPUs
    assert cluster.gpus_needed(75, 80) == 2


def test_gpus_needed_rounds_3_up_to_4():
    assert cluster.gpus_needed(200, 80) == 4


def test_gpus_needed_above_8_uses_multiples_of_8():
    assert cluster.gpus_needed(700, 80) == 16


def test_gpus_needed_bad_usable_raises():
    with pytest.raises(ValueError):
        cluster.gpus_needed(100, 80, usable=0)


@pytest.mark.parametrize("gpus, expected", [(1, 1), (8, 1), (9, 2), (16, 2), (24, 3)])
def test_servers_needed(gpus, expected):
    assert cluster.servers_needed(gpus) == expected


def test_servers_needed_custom_gpus_per_server():
    assert cluster.servers_needed(8, gpus_per_server=4) == 2


@pytest.mark.parametrize("mode, expected", [("none", 3), ("n+1", 4), ("N+1", 4), ("double", 6)])
def test_apply_backup(mode, expected):
    assert cluster.apply_backup(3, mode) == expected


def test_apply_backup_unknown_mode_raises():
    with pytest.raises(ValueError):
        cluster.apply_backup(3, "triple")
