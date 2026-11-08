"""One-page PDF summary of a sizing result (fpdf2). Plain ASCII only, returned as bytes."""

from fpdf import FPDF

from sizing.explain import explain, num
from ui.checks import result_warnings
from ui.display import assumption_note, network_rows, power_rows, speed_status, storage_rows

TITLE = "AI Infrastructure Sizing Summary"
FOOTER = "Indicative only, no prices."
ASCII_SWAPS = {"×": "x", "→": "->", "÷": "/", "—": "-", "–": "-", "≈": "~", "≥": ">=",
               "≤": "<=", "’": "'", "‘": "'", "“": '"', "”": '"', "…": "..."}


def to_ascii(text):
    """Swap symbols the default PDF fonts can't print for plain ASCII ('×' -> 'x', unknown -> '?')."""
    for symbol, plain in ASCII_SWAPS.items():
        text = text.replace(symbol, plain)
    return text.encode("ascii", "replace").decode("ascii")


def input_lines(r):
    """Return the inputs as 'name: value' lines."""
    inp, model = r["inputs"], r["model"]
    return [f"Model: {model['name']} ({num(model['params_b'])}B parameters)", f"Number format: {inp['precision']}",
            f"Users: {num(inp['users'], 0)}", f"Tokens per user: {num(inp['tokens_per_user'], 0)}",
            f"Target speed: {num(inp['target_tokens_per_s'])} tokens/s per user",
            f"GPU: {r['gpu']['name']}   Server: {r['server']['name']}   Backup: {inp['backup_mode']}",
            f"Rack power limit: {num(inp['rack_kw'])} kW   Load time target: {num(inp['load_time_s'])} s"]


def result_lines(r):
    """Return the result numbers as lines."""
    m, sp = r["memory"], r["speed"]
    rows = storage_rows(r) + network_rows(r) + power_rows(r)
    return [f"GPU memory: weights {num(m['weights_gb'])} GB + KV cache {num(m['kv_gb'])} GB "
            f"+ extra {num(m['extra_gb'])} GB = {num(m['total_gb'])} GB",
            f"GPUs: {r['gpus']['count']} x {r['gpu']['name']}",
            f"Servers: {r['servers']['after_backup']} (before backup: {r['servers']['before_backup']})",
            f"Speed: {num(sp['tokens_per_s_per_user'])} tokens/s per user - {speed_status(r)} "
            f"(target {num(sp['check']['target'])})"] + [f"{name}: {value}" for name, value in rows]


def assumption_lines(r):
    """Return the assumptions note and every setting used."""
    used = ", ".join(f"{name} = {value}" for name, value in r["settings"].items())
    return [assumption_note(r["settings"]), f"Settings used: {used}"]


def sections(r):
    """Return (heading, lines) pairs for the PDF; Warnings only if there are any."""
    steps = [f"{i}. {s['title']}: {s['filled_in']}" for i, s in enumerate(explain(r), start=1)]
    parts = [("Inputs", input_lines(r)), ("Results", result_lines(r)), ("Warnings", result_warnings(r)),
             ("Assumptions", assumption_lines(r)), ("How each number was found", steps)]
    return [(heading, [to_ascii(line) for line in lines]) for heading, lines in parts if lines]


class SummaryPDF(FPDF):
    """An A4 page with the 'Indicative only, no prices' footer."""

    def footer(self):
        """Print the footer at the bottom of every page."""
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, FOOTER, align="C")


def write_section(pdf, heading, lines):
    """Write one heading and its lines (long lines wrap)."""
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, heading, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for line in lines:
        pdf.multi_cell(0, 3.8, line, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)


def make_pdf(r):
    """Return the filled-in PDF object for this result."""
    pdf = SummaryPDF(format="A4")
    pdf.set_margins(15, 12, 15)
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, f"{TITLE} (Indicative)", new_x="LMARGIN", new_y="NEXT")
    for heading, lines in sections(r):
        write_section(pdf, heading, lines)
    return pdf


def build_pdf(r):
    """Return the PDF summary as bytes, ready for a download button."""
    return bytes(make_pdf(r).output())
