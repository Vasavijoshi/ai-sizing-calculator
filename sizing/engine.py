"""The whole sizing engine in one call: size_workload(). All results are indicative."""

from dataclasses import asdict, dataclass

from sizing import cluster, memory, power, storage_network, throughput
from sizing.defaults import DEFAULTS, gpu_spec, settings
from sizing.models import find_by_name, load_models, require_fields

MODEL_FIELDS = ["params_b", "layers", "kv_heads", "head_size"]


@dataclass
class Inputs:
    """What the user asks for: the workload, the GPU, and the backup choice."""
    model_name: str
    users: int
    tokens_per_user: int
    precision: str = "FP16"
    target_tokens_per_s: float = DEFAULTS["target_tokens_per_s"]
    gpu_name: str = "H100 SXM"
    backup_mode: str = "none"
    rack_kw: float = DEFAULTS["rack_kw"]
    load_time_s: float = DEFAULTS["load_time_s"]


def resolve_settings(inputs, overrides):
    """Return the settings to use: defaults, then overrides, then the user's inputs."""
    s = settings(overrides)
    s["target_tokens_per_s"] = inputs.target_tokens_per_s
    s["rack_kw"] = inputs.rack_kw
    s["load_time_s"] = inputs.load_time_s
    return s


def check_names_match(inputs, gpu, model):
    """Raise ValueError if the GPU or model rows don't match the names in inputs."""
    if gpu["name"] != inputs.gpu_name:
        raise ValueError(f"inputs ask for '{inputs.gpu_name}' but the GPU given is '{gpu['name']}'")
    if model["name"] != inputs.model_name:
        raise ValueError(f"inputs ask for '{inputs.model_name}' but the model given is '{model['name']}'")


def memory_part(model, inputs, s):
    """Return the memory breakdown: weights, KV cache, extra and total (GB)."""
    require_fields(model, MODEL_FIELDS)
    bpp = memory.bytes_per_param(inputs.precision)
    weights = memory.weights_gb(model["params_b"], bpp)
    kv_token = memory.kv_bytes_per_token(model["layers"], model["kv_heads"], model["head_size"], bpp)
    kv = memory.kv_total_gb(kv_token, inputs.tokens_per_user, inputs.users)
    total = memory.total_gpu_memory_gb(weights, kv, s["overhead"])
    return {"bytes_per_param": bpp, "kv_bytes_per_token": kv_token, "weights_gb": weights,
            "kv_gb": kv, "extra_gb": total - weights - kv, "total_gb": total}


def gpu_part(total_gb, gpu, server, s):
    """Return raw and rounded GPU counts, and whether the model spans servers."""
    gpu_memory = gpu_spec(gpu, "memory_gb")
    raw = cluster.raw_gpus(total_gb, gpu_memory, s["usable"])
    return {"raw": raw, "count": cluster.round_up_gpu_count(raw),
            "usable_gb_per_gpu": gpu_memory * s["usable"],
            "spans_servers": raw > server["gpus_per_server"]}


def server_part(gpus, server, backup_mode):
    """Return the server count before and after adding backup servers."""
    before = cluster.servers_needed(gpus, server["gpus_per_server"])
    return {"before_backup": before, "backup_mode": backup_mode,
            "after_backup": cluster.apply_backup(before, backup_mode)}


def speed_part(mem, gpus, gpu, s):
    """Return time per token, tokens per second per user, and the speed check."""
    bandwidth = gpu_spec(gpu, "bandwidth_gb_s")
    t = throughput.time_per_token_s(mem["weights_gb"], mem["kv_gb"], gpus, bandwidth,
                                    s["bandwidth_efficiency"])
    tps = throughput.tokens_per_s_per_user(t)
    return {"time_per_token_s": t, "tokens_per_s_per_user": tps,
            "check": throughput.speed_check(tps, s["target_tokens_per_s"])}


def storage_part(weights_gb, servers, s):
    """Return disk size (GB) and the read speed (GB/s) to load every server."""
    return {"size_gb": storage_network.storage_size_gb(weights_gb, s["model_versions"], s["log_extra"]),
            "read_gb_s": storage_network.storage_read_gb_s(weights_gb, servers, s["load_time_s"])}


def network_part(users, tokens_per_s, servers, spans_servers, s):
    """Return user traffic (Mb/s) and the network advice text."""
    traffic = storage_network.user_traffic_mbps(users, tokens_per_s, s["traffic_bytes_per_token"])
    return {"user_traffic_mbps": traffic,
            "advice": storage_network.network_advice(servers, spans_servers)}


def power_part(servers, gpu, server, s):
    """Return per-server kW, total kW, cooling (BTU/hr) and racks."""
    per_server = power.server_power_kw(server["gpus_per_server"], gpu_spec(gpu, "tdp_w"),
                                       s["server_power_factor"])
    total = power.total_power_kw(servers, per_server)
    return {"per_server_kw": per_server, "total_kw": total,
            "btu_per_hr": power.cooling_btu_per_hr(total),
            "racks": power.racks_needed(servers, per_server, s["rack_kw"])}


def size_workload(inputs, gpu, server, overrides=None, model=None):
    """Run every calculation in order and return one plain dict of indicative results."""
    s = resolve_settings(inputs, overrides)
    model = model or find_by_name(load_models(), inputs.model_name)
    check_names_match(inputs, gpu, model)
    mem = memory_part(model, inputs, s)
    gpus = gpu_part(mem["total_gb"], gpu, server, s)
    servers = server_part(gpus["count"], server, inputs.backup_mode)
    speed = speed_part(mem, gpus["count"], gpu, s)
    n = servers["after_backup"]
    return {"indicative": True, "inputs": asdict(inputs), "settings": s,
            "model": dict(model), "gpu": dict(gpu), "server": dict(server),
            "memory": mem, "gpus": gpus, "servers": servers, "speed": speed,
            "storage": storage_part(mem["weights_gb"], n, s),
            "network": network_part(inputs.users, speed["tokens_per_s_per_user"], n, gpus["spans_servers"], s),
            "power": power_part(n, gpu, server, s)}
