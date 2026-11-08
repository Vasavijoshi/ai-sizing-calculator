"""Tests for the one-page PDF summary."""

from conftest import FAKE_BIG_MODEL
from exports import pdf_summary


def all_text(result):
    """Return every line that goes into the PDF, joined into one string."""
    return "\n".join(line for _, lines in pdf_summary.sections(result) for line in lines)


def test_to_ascii_swaps_symbols():
    assert pdf_summary.to_ascii("8 × 700 W ÷ 1000 → 5.6 kW — about ≈") == "8 x 700 W / 1000 -> 5.6 kW - about ~"


def test_to_ascii_replaces_unknown_characters():
    assert pdf_summary.to_ascii("GPU 🚀 模型") == "GPU ? ??"


def test_pdf_bytes_start_with_pdf_marker(run):
    assert pdf_summary.build_pdf(run()).startswith(b"%PDF")


def test_worked_example_fits_on_one_page(run):
    assert pdf_summary.make_pdf(run()).pages_count == 1


def test_sections_hold_inputs_results_and_assumptions(run):
    headings = [heading for heading, _ in pdf_summary.sections(run())]
    assert headings == ["Inputs", "Results", "Warnings", "Assumptions", "How each number was found"]
    text = all_text(run())
    for words in ["Users: 200", "Tokens per user: 4,096", "GPUs: 8 x H100 SXM",
                  "39.4 tokens/s per user - PASS", "8.4 kW", "overhead = 1.15"]:
        assert words in text


def test_all_pdf_text_is_plain_ascii(run):
    assert all_text(run(backup_mode="n+1")).isascii()


def test_build_never_crashes_on_explain_text(run):
    cases = [run(), run(backup_mode="double"), run(target_tokens_per_s=50),
             run(users=1, tokens_per_user=130_000),
             run(model=FAKE_BIG_MODEL, model_name=FAKE_BIG_MODEL["name"])]
    for result in cases:
        assert pdf_summary.build_pdf(result).startswith(b"%PDF")


def test_no_warnings_section_when_quiet(run):
    headings = [heading for heading, _ in pdf_summary.sections(run(users=1))]
    assert "Warnings" not in headings


def test_footer_text():
    assert pdf_summary.FOOTER == "Indicative only, no prices."
