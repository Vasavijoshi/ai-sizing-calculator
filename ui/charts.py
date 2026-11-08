"""The GPU memory chart: what is needed (stacked) next to what the GPUs can hold."""

import plotly.graph_objects as go

COLOURS = {"Weights": "#4C78A8", "KV cache": "#F58518", "Extra": "#B0B0B0", "Usable memory": "#54A24B"}


def memory_chart_data(result):
    """Return the chart numbers: needed parts (GB) and the usable memory of the chosen GPUs."""
    m, g = result["memory"], result["gpus"]
    return {"needed": [("Weights", m["weights_gb"]), ("KV cache", m["kv_gb"]), ("Extra", m["extra_gb"])],
            "usable_gb": g["count"] * g["usable_gb_per_gpu"],
            "usable_label": f"Usable on {g['count']} x {result['gpu']['name']}"}


def bar(x_label, name, value):
    """Return one Plotly bar with a GB label on it."""
    return go.Bar(x=[x_label], y=[value], name=name, marker_color=COLOURS[name],
                  text=[f"{value:,.1f} GB"], textposition="inside")


def memory_figure(result):
    """Return a stacked bar of needed memory next to a bar of usable GPU memory."""
    data = memory_chart_data(result)
    bars = [bar("Needed", name, value) for name, value in data["needed"]]
    bars.append(bar(data["usable_label"], "Usable memory", data["usable_gb"]))
    fig = go.Figure(bars)
    fig.update_layout(barmode="stack", yaxis_title="GB", height=380,
                      margin={"t": 30, "b": 30}, legend={"orientation": "h"})
    return fig
