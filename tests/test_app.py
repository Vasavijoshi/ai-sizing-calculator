"""Smoke tests: start the real Streamlit app headless and check the screen."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def metrics(at):
    """Return the summary cards as {label: value}."""
    return {m.label: m.value for m in at.metric}


def widget(elements, label):
    """Return the widget with this label."""
    return next(e for e in elements if e.label == label)


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_starts_with_defaults_and_no_exception(app):
    assert not app.exception
    assert metrics(app) == {"GPUs": "8", "Servers": "1", "Speed per user": "39.4",
                            "Total power": "8.4 kW"}
    assert any("PASS" in m.value for m in app.markdown)


def test_indicative_note_is_shown(app):
    assert any(c.value.startswith("Indicative") for c in app.caption)


def test_switch_to_h200(app):
    widget(app.selectbox, "GPU").set_value("H200 SXM").run()
    assert not app.exception
    assert metrics(app)["GPUs"] == "4"
    assert metrics(app)["Speed per user"] == "28.2"


def test_zero_users_shows_friendly_message(app):
    widget(app.number_input, "Users").set_value(0).run()
    assert not app.exception
    assert any("Users must be at least 1" in i.value for i in app.info)


def test_about_tab_shows_real_default_numbers(app):
    from sizing.defaults import DEFAULTS
    from sizing.explain import num
    about_tab = widget(app.tabs, "About and limits")
    text = " ".join(m.value for m in about_tab.markdown)
    for value in DEFAULTS.values():
        assert f"**{num(value, 2)}**" in text
    assert "Indicative only: confirm every design with the vendor's sizing tools." in text


def test_about_tab_follows_the_sidebar(app):
    widget(app.number_input, "Target tokens/s per user").set_value(50.0).run()
    text = " ".join(m.value for m in widget(app.tabs, "About and limits").markdown)
    assert "at least **50** tokens per second" in text


def test_about_tab_still_shows_when_inputs_are_blocked(app):
    widget(app.number_input, "Users").set_value(0).run()
    assert "What this tool does" in " ".join(m.value for m in widget(app.tabs, "About and limits").markdown)


def test_compare_tab_table(app):
    table = app.dataframe[0].value
    assert list(table["GPU"]) == ["H100 SXM", "H200 SXM"]
    assert list(table["GPUs"]) == [8, 4]
