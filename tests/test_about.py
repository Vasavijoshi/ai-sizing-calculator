"""Tests for the 'About and limits' text."""

from sizing.defaults import DEFAULTS
from sizing.explain import num
from ui import about


def test_every_default_has_a_plain_words_line():
    assert set(about.ASSUMPTION_WORDS) == set(DEFAULTS)


def test_assumption_lines_show_the_real_default_numbers():
    text = "\n".join(about.assumption_lines(DEFAULTS))
    for name, value in DEFAULTS.items():
        assert f"**{num(value, 2)}**" in text, name


def test_assumption_lines_follow_changed_settings():
    changed = dict(DEFAULTS, overhead=1.3)
    assert "**1.3**" in "\n".join(about.assumption_lines(changed))


def test_limits_cover_what_the_tool_does_not_do():
    text = " ".join(about.LIMITS).lower()
    for words in ["prices", "training", "benchmarks", "prompt"]:
        assert words in text


def test_full_text_has_all_parts_and_the_disclaimer():
    text = about.about_markdown(DEFAULTS)
    for heading in ["What this tool does", "Assumptions it uses", "What it does NOT do"]:
        assert heading in text
    assert "Indicative only: confirm every design with the vendor's sizing tools." in text
