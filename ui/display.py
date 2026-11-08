"""Turn an engine result into short text for the screen (cards, sections, notes)."""

from sizing.explain import num


def speed_status(result):
    """Return 'PASS' or 'FAIL' for the speed check."""
    return "PASS" if result["speed"]["check"]["passes"] else "FAIL"


def summary_cards(result):
    """Return the four summary cards as dicts with label, value, note and status."""
    sp, sv = result["speed"], result["servers"]
    return [
        {"label": "GPUs", "value": str(result["gpus"]["count"]), "note": result["gpu"]["name"], "status": None},
        {"label": "Servers", "value": str(sv["after_backup"]), "note": f"backup: {sv['backup_mode']}", "status": None},
        {"label": "Speed per user", "value": num(sp["tokens_per_s_per_user"]),
         "note": f"tokens/s, target {num(sp['check']['target'])}", "status": speed_status(result)},
        {"label": "Total power", "value": f"{num(result['power']['total_kw'])} kW", "note": "indicative", "status": None},
    ]


def storage_rows(result):
    """Return (name, value) rows for the Storage section."""
    st = result["storage"]
    return [("Disk size", f"{num(st['size_gb'])} GB"), ("Read speed", f"{num(st['read_gb_s'], 2)} GB/s")]


def network_rows(result):
    """Return (name, value) rows for the Network section."""
    nw = result["network"]
    return [("User traffic", f"{num(nw['user_traffic_mbps'], 2)} Mb/s"), ("Advice", nw["advice"])]


def power_rows(result):
    """Return (name, value) rows for the Power section."""
    p = result["power"]
    return [("Per server", f"{num(p['per_server_kw'])} kW"), ("Total", f"{num(p['total_kw'])} kW"),
            ("Cooling", f"{num(p['btu_per_hr'], 0)} BTU/hr"), ("Racks", str(p["racks"]))]


def step_markdown(number, step):
    """Return one 'Why this number' step as Markdown text."""
    return (f"**{number}. {step['title']}**  \n"
            f"Formula: {step['formula']}  \n"
            f"Filled in: {step['filled_in']}  \n"
            f"Result: **{step['result']}**  \n"
            f"Assumptions: {' '.join(step['assumptions'])}")


def assumption_note(s):
    """Return the one-line 'Indicative' note naming the main assumptions (s = settings)."""
    return (f"Indicative only, no prices. Assumes {num((s['overhead'] - 1) * 100, 0)}% extra memory, "
            f"{num(s['usable'] * 100, 0)}% of each GPU's memory used, "
            f"{num(s['bandwidth_efficiency'] * 100, 0)}% of peak memory speed, "
            "and every user at full conversation length at once.")
