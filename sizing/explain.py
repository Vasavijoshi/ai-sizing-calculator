"""Turn an engine result into simple 'why this number' steps. All numbers are indicative."""

import math


def num(x, digits=1):
    """Format a number for people: commas, no trailing zeros (140.0 -> '140')."""
    text = f"{x:,.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def step(title, formula, filled_in, result, assumptions):
    """Return one explanation step with the 5 standard keys."""
    return {"title": title, "formula": formula, "filled_in": filled_in,
            "result": result, "assumptions": assumptions}


def weights_step(r):
    """Explain the model weights size."""
    m, model = r["memory"], r["model"]
    return step("Model weights",
                "weights (GB) = parameters (billions) × bytes per parameter",
                f"{num(model['params_b'])} billion × {num(m['bytes_per_param'])} bytes = {num(m['weights_gb'])} GB",
                f"{num(m['weights_gb'])} GB",
                [f"{r['inputs']['precision']} stores each number in {num(m['bytes_per_param'])} bytes.",
                 "1 GB = 1,000,000,000 bytes."])


def kv_token_step(r):
    """Explain the KV cache size for one token."""
    m, model = r["memory"], r["model"]
    return step("KV cache per token",
                "KV per token = 2 (one key + one value) × layers × KV heads × head size × bytes per value",
                f"2 × {model['layers']} layers × {model['kv_heads']} KV heads × {model['head_size']} "
                f"× {num(m['bytes_per_param'])} bytes = {num(m['kv_bytes_per_token'], 0)} bytes",
                f"{num(m['kv_bytes_per_token'], 0)} bytes",
                [f"The KV cache uses the same precision as the weights ({r['inputs']['precision']})."])


def kv_total_step(r):
    """Explain the KV cache size for all users."""
    m, inp = r["memory"], r["inputs"]
    return step("KV cache for all users",
                "KV total (GB) = KV per token × tokens per user × users ÷ 1,000,000,000",
                f"{num(m['kv_bytes_per_token'], 0)} × {num(inp['tokens_per_user'], 0)} tokens "
                f"× {num(inp['users'], 0)} users ÷ 1,000,000,000 = {num(m['kv_gb'])} GB",
                f"{num(m['kv_gb'])} GB",
                ["Worst case: every user has a full-length conversation at the same time."])


def total_memory_step(r):
    """Explain the total GPU memory with the extra room."""
    m, overhead = r["memory"], r["settings"]["overhead"]
    return step("Total GPU memory",
                "total (GB) = (weights + KV cache) × overhead",
                f"({num(m['weights_gb'])} + {num(m['kv_gb'])}) × {num(overhead, 2)} = {num(m['total_gb'])} GB",
                f"{num(m['total_gb'])} GB",
                [f"overhead = {num(overhead, 2)}: about {num((overhead - 1) * 100, 0)}% extra room "
                 "for the software, working memory and wasted gaps."])


def gpu_step(r):
    """Explain how the GPU count was found and rounded."""
    m, g, gpu, usable = r["memory"], r["gpus"], r["gpu"], r["settings"]["usable"]
    return step("GPUs",
                "GPUs = total ÷ (GPU memory × usable share), rounded up to 1, 2, 4 or 8 (above 8: a multiple of 8)",
                f"{num(m['total_gb'])} ÷ ({num(gpu['memory_gb'])} × {num(usable, 2)} = {num(g['usable_gb_per_gpu'])} GB) "
                f"= {num(m['total_gb'] / g['usable_gb_per_gpu'], 2)} → {g['raw']} → {g['count']} GPUs",
                f"{g['count']} × {gpu['name']}",
                [f"usable = {num(usable, 2)}: we only plan to fill {num(usable * 100, 0)}% of each GPU.",
                 "A model is usually split across 1, 2, 4 or 8 GPUs."])


def server_step(r):
    """Explain the server count and the backup servers."""
    g, sv, server = r["gpus"], r["servers"], r["server"]
    return step("Servers",
                "servers = GPUs ÷ GPUs per server (rounded up), then add backup",
                f"{g['count']} ÷ {server['gpus_per_server']} = {sv['before_backup']}; "
                f"backup '{sv['backup_mode']}' → {sv['after_backup']}",
                f"{sv['after_backup']} servers",
                [f"{server['name']} holds {server['gpus_per_server']} GPUs.",
                 "Backup: 'none' adds nothing, 'n+1' adds one spare, 'double' doubles."])


def speed_step(r):
    """Explain the time per token and the speed each user gets."""
    m, sp, gpu, s = r["memory"], r["speed"], r["gpu"], r["settings"]
    verdict = "passes" if sp["check"]["passes"] else "misses"
    return step("Speed per user",
                "time per token = (weights + KV) ÷ (GPUs × memory speed × efficiency); speed = 1 ÷ time",
                f"({num(m['weights_gb'])} + {num(m['kv_gb'])}) ÷ ({r['gpus']['count']} × "
                f"{num(gpu['bandwidth_gb_s'], 0)} GB/s × {num(s['bandwidth_efficiency'], 2)}) = "
                f"{num(sp['time_per_token_s'], 4)} s → 1 ÷ {num(sp['time_per_token_s'], 4)} "
                f"= {num(sp['tokens_per_s_per_user'])} tokens/s",
                f"{num(sp['tokens_per_s_per_user'])} tokens/s per user ({verdict} the target of "
                f"{num(s['target_tokens_per_s'])})",
                [f"efficiency = {num(s['bandwidth_efficiency'], 2)}: GPUs reach about "
                 f"{num(s['bandwidth_efficiency'] * 100, 0)}% of their peak memory speed.",
                 "Worst case: each new token reads all the weights and every user's full KV cache."])


def storage_size_step(r):
    """Explain the disk space needed."""
    w, st, s = r["memory"]["weights_gb"], r["storage"], r["settings"]
    return step("Storage size",
                "disk (GB) = weights × versions kept × extra for logs",
                f"{num(w)} × {s['model_versions']} × {num(s['log_extra'], 2)} = {num(st['size_gb'])} GB",
                f"{num(st['size_gb'])} GB",
                [f"model_versions = {s['model_versions']}: we keep {s['model_versions']} versions of the weights.",
                 f"log_extra = {num(s['log_extra'], 2)}: extra space for logs and other files."])


def storage_read_step(r):
    """Explain the storage read speed needed to load the model."""
    w, st, s, n = r["memory"]["weights_gb"], r["storage"], r["settings"], r["servers"]["after_backup"]
    return step("Storage read speed",
                "read speed (GB/s) = weights × servers ÷ load time",
                f"{num(w)} × {n} servers ÷ {num(s['load_time_s'])} s = {num(st['read_gb_s'], 2)} GB/s",
                f"{num(st['read_gb_s'], 2)} GB/s",
                [f"load_time_s = {num(s['load_time_s'])}: every server loads its own copy in that time."])


def traffic_step(r):
    """Explain the network traffic to users."""
    inp, sp, nw, s = r["inputs"], r["speed"], r["network"], r["settings"]
    return step("User traffic",
                "traffic (Mb/s) = users × tokens per second × bytes per token × 8 ÷ 1,000,000",
                f"{num(inp['users'], 0)} users × {num(sp['tokens_per_s_per_user'])} tokens/s × "
                f"{s['traffic_bytes_per_token']} bytes × 8 ÷ 1,000,000 = {num(nw['user_traffic_mbps'], 2)} Mb/s",
                f"{num(nw['user_traffic_mbps'], 2)} Mb/s",
                [f"traffic_bytes_per_token = {s['traffic_bytes_per_token']}: bytes sent per token.",
                 "Uses the speed the system reaches, not the target."])


def network_step(r):
    """Explain the network advice."""
    n, spans = r["servers"]["after_backup"], r["gpus"]["spans_servers"]
    return step("Network",
                "rule of thumb: 1 server, several servers, or one model split across servers",
                f"{n} server(s); model split across servers: {'yes' if spans else 'no'}",
                r["network"]["advice"],
                ["Rule of thumb only. Check the design with a network engineer."])


def power_server_step(r):
    """Explain the power of one server."""
    p, gpu, server, s = r["power"], r["gpu"], r["server"], r["settings"]
    return step("Power per server",
                "server kW = GPU slots × GPU power (W) × server factor ÷ 1000",
                f"{server['gpus_per_server']} × {num(gpu['tdp_w'], 0)} W × {num(s['server_power_factor'], 2)} "
                f"÷ 1000 = {num(p['per_server_kw'])} kW",
                f"{num(p['per_server_kw'])} kW",
                ["Every GPU slot is counted as full.",
                 f"server_power_factor = {num(s['server_power_factor'], 2)}: adds CPUs, fans, network and so on."])


def power_total_step(r):
    """Explain the total power."""
    p, n = r["power"], r["servers"]["after_backup"]
    return step("Total power",
                "total kW = servers × server kW",
                f"{n} × {num(p['per_server_kw'])} kW = {num(p['total_kw'])} kW",
                f"{num(p['total_kw'])} kW",
                ["Backup servers are counted as switched on."])


def cooling_step(r):
    """Explain the cooling needed."""
    p = r["power"]
    return step("Cooling",
                "cooling (BTU/hr) = total kW × 3,412",
                f"{num(p['total_kw'])} × 3,412 = {num(p['btu_per_hr'], 0)} BTU/hr",
                f"{num(p['btu_per_hr'], 0)} BTU/hr",
                ["All electrical power turns into heat that must be removed."])


def racks_step(r):
    """Explain how many racks are needed."""
    p, n, rack_kw = r["power"], r["servers"]["after_backup"], r["settings"]["rack_kw"]
    per_rack = max(1, math.floor(rack_kw / p["per_server_kw"]))
    return step("Racks",
                "servers per rack = rack kW ÷ server kW (rounded down, at least 1); racks = servers ÷ that",
                f"{num(rack_kw)} kW ÷ {num(p['per_server_kw'])} kW → {per_rack} per rack; "
                f"{n} ÷ {per_rack} = {p['racks']} racks",
                f"{p['racks']} racks",
                [f"rack_kw = {num(rack_kw)}: the power one rack can supply and cool."])


STEPS = [weights_step, kv_token_step, kv_total_step, total_memory_step, gpu_step, server_step,
         speed_step, storage_size_step, storage_read_step, traffic_step, network_step,
         power_server_step, power_total_step, cooling_step, racks_step]


def explain(result):
    """Return one simple explanation step per calculation, in order."""
    return [make_step(result) for make_step in STEPS]
