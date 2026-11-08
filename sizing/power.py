"""Power, cooling and rack sizing. Results are indicative."""

import math

BTU_PER_HR_PER_KW = 3412


def server_power_kw(gpus_per_server, gpu_tdp_w, factor):
    """Return one server's power in kW, counting all its GPU slots as full."""
    return gpus_per_server * gpu_tdp_w * factor / 1000


def total_power_kw(servers, per_server_kw):
    """Return the power of all servers together, in kW."""
    return servers * per_server_kw


def cooling_btu_per_hr(kw):
    """Return the heat to remove in BTU per hour (1 kW = 3412 BTU/hr)."""
    return kw * BTU_PER_HR_PER_KW


def racks_needed(servers, per_server_kw, rack_kw):
    """Return how many racks are needed, given the power each rack can supply."""
    if per_server_kw <= 0:
        raise ValueError("per_server_kw must be above zero")
    servers_per_rack = max(1, math.floor(rack_kw / per_server_kw))
    return math.ceil(servers / servers_per_rack)
