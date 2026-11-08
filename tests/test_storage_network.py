"""Tests for storage and network sizing."""

import pytest

from sizing import storage_network as sn


def test_storage_size_worked_example():
    assert sn.storage_size_gb(140, 3, 1.2) == pytest.approx(504)


def test_storage_read_one_server():
    assert sn.storage_read_gb_s(140, 1, 120) == pytest.approx(1.17, abs=0.005)


def test_storage_read_two_servers():
    assert sn.storage_read_gb_s(140, 2, 120) == pytest.approx(2.33, abs=0.005)


def test_storage_read_zero_load_time_raises():
    with pytest.raises(ValueError):
        sn.storage_read_gb_s(140, 1, 0)


def test_user_traffic_mbps():
    # 200 users x 39.37 tokens/s x 10 bytes x 8 bits / 1e6 = about 0.63 Mb/s
    assert sn.user_traffic_mbps(200, 39.37, 10) == pytest.approx(0.63, abs=0.005)


def test_network_advice_one_server():
    assert sn.network_advice(1, False) == sn.ADVICE_ONE_SERVER


def test_network_advice_several_servers():
    assert sn.network_advice(2, False) == sn.ADVICE_SEVERAL_SERVERS


def test_network_advice_spans_servers_wins():
    assert sn.network_advice(2, True) == sn.ADVICE_SPANS_SERVERS


def test_advice_texts():
    assert "2 x 25 or 100 GbE" in sn.ADVICE_ONE_SERVER
    assert "100 GbE front end" in sn.ADVICE_SEVERAL_SERVERS
    assert "400 Gb/s" in sn.ADVICE_SPANS_SERVERS
