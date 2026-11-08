"""Plain-words text for the 'About and limits' tab. Numbers come from the settings, never retyped."""

from sizing.explain import num

DISCLAIMER = "Indicative only: confirm every design with the vendor's sizing tools."

WHAT_IT_DOES = (
    "This tool turns an AI serving workload (a model, a number of users and a conversation "
    "length) into a first estimate of GPU memory, GPUs, servers, speed per user, storage, "
    "network, power, cooling and racks. Every number comes with a 'Why this number' step "
    "so you can check how it was found."
)

# One sentence per default setting. {v} is filled in with the real value.
ASSUMPTION_WORDS = {
    "overhead": "Extra memory: weights plus KV cache are multiplied by {v} for software, working memory and wasted gaps.",
    "usable": "Usable memory: we only plan to fill {v} of each GPU's memory (1 = all of it).",
    "bandwidth_efficiency": "Speed efficiency: GPUs reach {v} of their peak memory speed (1 = full speed).",
    "server_power_factor": "Server power factor: a server uses {v} times the power of its GPUs (CPUs, fans, network).",
    "traffic_bytes_per_token": "User traffic: {v} bytes are sent to the user for each token.",
    "load_time_s": "Load time: every server should load the model from storage in {v} seconds.",
    "model_versions": "Storage: {v} versions of the model weights are kept on disk.",
    "log_extra": "Storage: disk space is multiplied by {v} to leave room for logs.",
    "rack_kw": "Racks: one rack can supply and cool {v} kW.",
    "target_tokens_per_s": "Speed target: each user should get at least {v} tokens per second.",
}

LIMITS = [
    "No prices or costs of any kind.",
    "No training sizing: it only sizes serving (inference) of a model that is already trained.",
    "No real benchmarks: speed comes from a simple memory-speed formula, not from measured runs.",
    "The prompt-reading phase (prefill) is ignored in the speed check; only answer writing is counted.",
    "It assumes the worst case: every user at full conversation length at the same time.",
    "Network advice is a rule of thumb, not a network design.",
]


def assumption_lines(settings):
    """Return one plain-words line per setting, with the real number in bold."""
    return [f"- {ASSUMPTION_WORDS[name].format(v=f'**{num(value, 2)}**')}"
            for name, value in settings.items() if name in ASSUMPTION_WORDS]


def about_markdown(settings):
    """Return the full 'About and limits' text as Markdown."""
    limits = "\n".join(f"- {line}" for line in LIMITS)
    return (f"#### What this tool does\n{WHAT_IT_DOES}\n\n"
            f"#### Assumptions it uses\nThese are the values used right now. The starting values come "
            f"from sizing/defaults.py; the first four can be changed under 'Advanced assumptions' in the sidebar.\n\n" + "\n".join(assumption_lines(settings)) +
            f"\n\n#### What it does NOT do\n{limits}\n\n**{DISCLAIMER}**")
