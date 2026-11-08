"""Tests for the card and section text shown on screen."""

from ui import display


def card(cards, label):
    """Return the card with this label."""
    return next(c for c in cards if c["label"] == label)


def test_default_cards(run):
    cards = display.summary_cards(run())
    assert card(cards, "GPUs")["value"] == "8"
    assert card(cards, "Servers")["value"] == "1"
    assert card(cards, "Speed per user")["value"] == "39.4"
    assert card(cards, "Speed per user")["status"] == "PASS"
    assert card(cards, "Total power")["value"] == "8.4 kW"


def test_h200_cards(run):
    cards = display.summary_cards(run(gpu_name="H200 SXM"))
    assert card(cards, "GPUs")["value"] == "4"
    assert card(cards, "Speed per user")["value"] == "28.2"


def test_fail_status(run):
    cards = display.summary_cards(run(target_tokens_per_s=50))
    assert card(cards, "Speed per user")["status"] == "FAIL"


def test_servers_card_shows_after_backup(run):
    assert card(display.summary_cards(run(backup_mode="n+1")), "Servers")["value"] == "2"


def test_section_rows(run):
    r = run()
    assert ("Disk size", "504 GB") in display.storage_rows(r)
    assert ("Read speed", "1.17 GB/s") in display.storage_rows(r)
    assert ("User traffic", "0.63 Mb/s") in display.network_rows(r)
    assert ("Cooling", "28,661 BTU/hr") in display.power_rows(r)
    assert ("Racks", "1") in display.power_rows(r)


def test_step_markdown_shows_all_parts():
    step = {"title": "Model weights", "formula": "weights = params x bytes",
            "filled_in": "70 billion × 2 bytes = 140 GB", "result": "140 GB",
            "assumptions": ["FP16 uses 2 bytes.", "1 GB = 1e9 bytes."]}
    text = display.step_markdown(1, step)
    assert "**1. Model weights**" in text
    for part in ["weights = params x bytes", "70 billion × 2 bytes = 140 GB", "**140 GB**",
                 "FP16 uses 2 bytes.", "1 GB = 1e9 bytes."]:
        assert part in text


def test_assumption_note_is_indicative_and_uses_settings(run):
    note = display.assumption_note(run(overrides={"overhead": 1.2})["settings"])
    assert note.startswith("Indicative")
    assert "20% extra memory" in note
    assert "90%" in note and "60%" in note
