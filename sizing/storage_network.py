"""Storage and network sizing. Results are indicative."""

ADVICE_ONE_SERVER = "2 x 25 or 100 GbE"
ADVICE_SEVERAL_SERVERS = "100 GbE front end plus a storage network"
# TODO: confirm the exact link speed for models that span servers.
ADVICE_SPANS_SERVERS = "400 Gb/s class links between servers (InfiniBand or RDMA Ethernet)"


def storage_size_gb(weights_gb, versions, log_extra):
    """Return disk space in GB: weights x number of versions kept x extra for logs."""
    return weights_gb * versions * log_extra


def storage_read_gb_s(weights_gb, copies, load_time_s):
    """Return storage read speed (GB/s) to load every server's copy in load_time_s."""
    if load_time_s <= 0:
        raise ValueError("load_time_s must be above zero")
    return weights_gb * copies / load_time_s


def user_traffic_mbps(users, tokens_per_s, bytes_per_token):
    """Return traffic to users in megabits per second (8 bits per byte)."""
    return users * tokens_per_s * bytes_per_token * 8 / 1e6


def network_advice(servers, spans_servers):
    """Return a short rule-of-thumb network suggestion."""
    if spans_servers:
        return ADVICE_SPANS_SERVERS
    if servers <= 1:
        return ADVICE_ONE_SERVER
    return ADVICE_SEVERAL_SERVERS
