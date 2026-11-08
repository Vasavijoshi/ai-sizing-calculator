"""Tests for the Compare GPUs table."""

import pytest

from conftest import worked_inputs
from sizing import models
from ui import compare


@pytest.fixture(scope="module")
def table(gpus, server):
    chosen = [models.find_by_name(gpus, name) for name in ["H100 SXM", "H200 SXM"]]
    return compare.compare_table(worked_inputs(), chosen, server)


def row(table, gpu_name):
    """Return the table row for this GPU."""
    return table[table["GPU"] == gpu_name].iloc[0]


def test_columns(table):
    assert list(table.columns) == compare.COLUMNS


def test_h100_row(table):
    r = row(table, "H100 SXM")
    assert r["GPUs"] == 8 and r["Servers"] == 1
    assert r["Speed (tokens/s per user)"] == pytest.approx(39.4, abs=0.05)
    assert r["Speed check"] == "PASS"


def test_h200_row(table):
    r = row(table, "H200 SXM")
    assert r["GPUs"] == 4 and r["Servers"] == 1
    assert r["Speed (tokens/s per user)"] == pytest.approx(28.2, abs=0.05)


def test_power_and_racks(table):
    r = row(table, "H100 SXM")
    assert r["Total power (kW)"] == pytest.approx(8.4)
    assert r["Racks"] == 1


def test_gpu_with_todo_field_gets_a_message_row(gpus, server):
    chosen = [models.find_by_name(gpus, name) for name in ["H100 SXM", "A100 80GB SXM"]]
    table = compare.compare_table(worked_inputs(), chosen, server)
    a100 = row(table, "A100 80GB SXM")
    assert "bandwidth_gb_s" in a100["Speed check"]
    assert len(table) == 2


def test_choice_count_rule():
    assert compare.choice_problem(["A"]) == "Pick 2 or 3 GPUs to compare."
    assert compare.choice_problem(["A", "B"]) is None
    assert compare.choice_problem(["A", "B", "C"]) is None
    assert compare.choice_problem(["A", "B", "C", "D"]) == "Pick 2 or 3 GPUs to compare."
