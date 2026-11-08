"""Tests for the GPU memory chart."""

import pytest

from ui import charts


def test_chart_data_worked_example(run):
    data = charts.memory_chart_data(run())
    parts = dict(data["needed"])
    assert list(parts) == ["Weights", "KV cache", "Extra"]
    assert parts["Weights"] == 140
    assert parts["KV cache"] == pytest.approx(268.4, abs=0.05)
    assert sum(parts.values()) == pytest.approx(469.7, abs=0.05)
    assert data["usable_gb"] == pytest.approx(8 * 72)


def test_usable_label_names_gpus(run):
    assert charts.memory_chart_data(run())["usable_label"] == "Usable on 8 x H100 SXM"


def test_figure_is_stacked_with_gb_axis(run):
    fig = charts.memory_figure(run())
    assert fig.layout.barmode == "stack"
    assert fig.layout.yaxis.title.text == "GB"
    assert [trace.name for trace in fig.data] == ["Weights", "KV cache", "Extra", "Usable memory"]


def test_usable_bar_sits_next_to_needed_bar(run):
    fig = charts.memory_figure(run())
    assert fig.data[0].x[0] == "Needed"
    assert fig.data[3].x[0] == "Usable on 8 x H100 SXM"
    assert fig.data[3].y[0] == pytest.approx(576)
