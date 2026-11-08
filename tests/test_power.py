"""Tests for power, cooling and racks."""

import pytest

from sizing import power


def test_server_power_worked_example():
    assert power.server_power_kw(8, 700, 1.5) == pytest.approx(8.4)


def test_total_power():
    assert power.total_power_kw(2, 8.4) == pytest.approx(16.8)


def test_cooling_btu_per_hr():
    assert power.cooling_btu_per_hr(8.4) == pytest.approx(28_661, abs=1)


@pytest.mark.parametrize("servers, per_server_kw, rack_kw, expected", [
    (1, 8.4, 15, 1),   # 1 server fits in a 15 kW rack
    (2, 8.4, 15, 2),   # only 1 server per rack (2 x 8.4 = 16.8 > 15)
    (4, 5, 15, 2),     # 3 servers per rack -> 2 racks
    (3, 20, 15, 3),    # server bigger than a rack -> still 1 per rack
    (0, 8.4, 15, 0),
])
def test_racks_needed(servers, per_server_kw, rack_kw, expected):
    assert power.racks_needed(servers, per_server_kw, rack_kw) == expected


def test_racks_needed_zero_server_power_raises():
    with pytest.raises(ValueError):
        power.racks_needed(1, 0, 15)
