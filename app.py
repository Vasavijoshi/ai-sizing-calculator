"""Streamlit screen for the AI Infrastructure Sizing Calculator. Run: python -m streamlit run app.py"""

import streamlit as st

from exports import pdf_summary
from sizing import models
from sizing.defaults import settings
from sizing.engine import resolve_settings
from sizing.explain import explain
from ui import about, charts, checks, compare, display, forms

st.set_page_config(page_title="AI Infrastructure Sizing Calculator", layout="wide")


@st.cache_data
def load_data():
    """Read the YAML data files once."""
    return models.load_models(), models.load_gpus(), models.load_servers()


def custom_model_fields():
    """Show the Custom model fields (empty to start) and return the custom model row."""
    sb = st.sidebar
    return forms.custom_model(
        params_b=sb.number_input("Parameters (billions)", min_value=0.1, value=None),
        layers=sb.number_input("Layers", min_value=1, value=None, step=1),
        kv_heads=sb.number_input("KV heads", min_value=1, value=None, step=1),
        head_size=sb.number_input("Head size", min_value=1, value=None, step=1))


def model_picker(model_rows):
    """Show the model dropdown and return the chosen model row."""
    choice = st.sidebar.selectbox("Model", forms.model_choices(model_rows))
    st.sidebar.caption(forms.needs_config_caption(forms.incomplete_models(model_rows)))
    custom = custom_model_fields() if choice == forms.CUSTOM else None
    return forms.pick_model(model_rows, choice, custom)


def gpu_picker(gpus):
    """Show the GPU dropdown and return the chosen GPU row."""
    names = forms.gpu_choices(gpus)
    start = names.index("H100 SXM") if "H100 SXM" in names else 0
    name = st.sidebar.selectbox("GPU", names, index=start)
    st.sidebar.caption(forms.needs_config_caption(forms.incomplete_gpus(gpus)))
    return models.find_by_name(gpus, name)


def advanced_sliders():
    """Show the 'Advanced assumptions' sliders and return them as overrides."""
    with st.sidebar.expander("Advanced assumptions"):
        values = {name: st.slider(cfg["label"], cfg["min"], cfg["max"], forms.slider_start(name), cfg["step"])
                  for name, cfg in forms.SLIDERS.items()}
    return forms.build_overrides(values)


def workload_numbers():
    """Show the users, tokens and target fields; return (users, tokens, target)."""
    sb = st.sidebar
    users = sb.number_input("Users", min_value=0, max_value=forms.USERS_MAX, value=forms.START_USERS, step=1)
    tokens = sb.number_input("Tokens per user", min_value=0, value=forms.START_TOKENS, step=256)
    target = sb.number_input("Target tokens/s per user", min_value=1.0, value=float(forms.DEFAULTS["target_tokens_per_s"]))
    return users, tokens, target


def sidebar_form(model_rows, gpus, servers):
    """Show the whole sidebar form; return (inputs, model, gpu, server, overrides)."""
    sb = st.sidebar
    sb.header("Workload")
    model = model_picker(model_rows)
    precision = sb.selectbox("Number format", list(forms.PRECISION_OPTIONS))
    users, tokens, target = workload_numbers()
    gpu = gpu_picker(gpus)
    server = models.find_by_name(servers, sb.selectbox("Server", [s["name"] for s in servers]))
    backup = sb.selectbox("Backup", forms.BACKUP_OPTIONS)
    rack_kw = sb.number_input("Rack power limit (kW)", min_value=1.0, value=float(forms.DEFAULTS["rack_kw"]))
    load_time = sb.number_input("Load time target (s)", min_value=1.0, value=float(forms.DEFAULTS["load_time_s"]))
    inputs = forms.build_inputs(model["name"], precision, users, tokens, target, gpu["name"], backup, rack_kw, load_time)
    return inputs, model, gpu, server, advanced_sliders()


def show_cards(result):
    """Show the four summary cards."""
    for column, card in zip(st.columns(4), display.summary_cards(result)):
        column.metric(card["label"], card["value"])
        if card["status"]:
            colour = "green" if card["status"] == "PASS" else "red"
            column.markdown(f":{colour}-badge[{card['status']}] {card['note']}")
        else:
            column.caption(card["note"])


def show_rows(column, title, rows):
    """Show one small section as a title and 'name: value' lines."""
    column.subheader(title)
    for name, value in rows:
        column.markdown(f"**{name}:** {value}")


def show_sections(result):
    """Show the Storage, Network and Power sections side by side."""
    storage, network, power = st.columns(3)
    show_rows(storage, "Storage", display.storage_rows(result))
    show_rows(network, "Network", display.network_rows(result))
    show_rows(power, "Power", display.power_rows(result))


def show_explanation(result):
    """Show every 'Why this number' step inside an expander."""
    with st.expander("Why this number"):
        for number, step in enumerate(explain(result), start=1):
            st.markdown(display.step_markdown(number, step))


def show_indicative(title, settings):
    """Show a heading with the 'Indicative' badge and the one-line assumptions note."""
    st.markdown(f"### {title} :orange-badge[Indicative]")
    st.caption(display.assumption_note(settings))


def show_results(result):
    """Show the indicative note, warnings, cards, chart, sections and explanation."""
    show_indicative("Results", result["settings"])
    for message in checks.result_warnings(result):
        st.warning(message)
    show_cards(result)
    st.subheader("GPU memory")
    st.plotly_chart(charts.memory_figure(result), width="stretch")
    show_sections(result)
    show_explanation(result)
    st.download_button("Download PDF summary", data=pdf_summary.build_pdf(result),
                       file_name="sizing-summary.pdf", mime="application/pdf")


def show_unless_blocked(problems, show_tab):
    """Show the friendly 'blocked' messages if there are any, otherwise show the tab."""
    if not problems:
        show_tab()
        return
    for message in problems:
        st.info(message)


def results_tab(inputs, model, gpu, server, overrides):
    """Run the engine safely for the chosen GPU and show the results."""
    result, error = checks.safe_size(inputs, gpu, server, overrides=overrides, model=model)
    if error:
        st.error(error)
        return
    show_results(result)


def compare_tab(inputs, model, gpus, server, overrides):
    """Let the user pick 2 or 3 GPUs and show one comparison table."""
    show_indicative("Compare GPUs", settings(overrides))
    names = forms.gpu_choices(gpus)
    picked = st.multiselect("GPUs to compare (2 or 3)", names, default=names[:3])
    st.caption(forms.needs_config_caption(forms.incomplete_gpus(gpus)))
    problem = compare.choice_problem(picked)
    if problem:
        st.info(problem)
        return
    rows = [models.find_by_name(gpus, name) for name in picked]
    st.dataframe(compare.compare_table(inputs, rows, server, overrides, model), hide_index=True)


def main():
    """Collect inputs, check them, and show the Results and Compare GPUs tabs."""
    st.title("AI Infrastructure Sizing Calculator")
    model_rows, gpus, servers = load_data()
    inputs, model, gpu, server, overrides = sidebar_form(model_rows, gpus, servers)
    problems = checks.input_problems(inputs.users, inputs.tokens_per_user) + checks.custom_model_problems(model)
    results, comparison, about_tab = st.tabs(["Results", "Compare GPUs", "About and limits"])
    with results:
        show_unless_blocked(problems, lambda: results_tab(inputs, model, gpu, server, overrides))
    with comparison:
        show_unless_blocked(problems, lambda: compare_tab(inputs, model, gpus, server, overrides))
    with about_tab:
        st.markdown(about.about_markdown(resolve_settings(inputs, overrides)))


main()
