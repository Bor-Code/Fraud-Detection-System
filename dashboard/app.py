import html
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
REPORT_DIR = Path(os.environ.get("REPORT_DIR", "reports"))
REQUEST_TIMEOUT = 120
LABEL_COLUMN = "Class"
TABLE_ROW_LIMIT = 1000
BAND_EDGES = [-0.001, 0.2, 0.5, 1.0]
BAND_LABELS = ["Low (below 0.20)", "Elevated (0.20 to 0.50)", "High (above 0.50)"]

INK = "#14181F"
MUTED = "#5B6573"
RULE = "#D9DDE3"
SURFACE = "#FFFFFF"
CANVAS = "#F4F5F7"
ACCENT = "#1F4E79"
RISK = "#B42318"
STABLE = "#1B7F5A"

st.set_page_config(
    page_title="Fraud Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    color-scheme: light;
    --ink: __INK__;
    --muted: __MUTED__;
    --rule: __RULE__;
    --surface: __SURFACE__;
    --canvas: __CANVAS__;
    --accent: __ACCENT__;
    --risk: __RISK__;
    --stable: __STABLE__;
}

html, body, .stApp, [class*="css"] {
    font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--canvas);
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1280px;
    padding-top: 2.25rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4, p, label, li {
    color: var(--ink);
}

[data-testid="stWidgetLabel"] p {
    color: var(--ink);
    font-size: 0.9rem;
    font-weight: 500;
}

.app-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 2rem;
    border-bottom: 1px solid var(--rule);
    padding-bottom: 1.25rem;
    margin-bottom: 2rem;
}

.app-title {
    font-size: 1.75rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.2;
}

.app-subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0.35rem 0 0 0;
    max-width: 62ch;
}

.service {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.9rem;
    color: var(--muted);
    white-space: nowrap;
    padding-bottom: 0.2rem;
}

.dot {
    width: 0.55rem;
    height: 0.55rem;
    border-radius: 50%;
    display: inline-block;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 0.25rem 0;
}

.section-note {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0 0 1rem 0;
    max-width: 70ch;
}

.strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 4px;
    margin: 1rem 0 1.25rem 0;
}

.strip-cell {
    padding: 1rem 1.25rem;
    border-left: 1px solid var(--rule);
}

.strip-cell:first-child {
    border-left: none;
}

.strip-label {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0 0 0.3rem 0;
}

.strip-value {
    font-size: 1.65rem;
    font-weight: 600;
    line-height: 1.15;
    margin: 0;
    font-variant-numeric: tabular-nums;
}

.strip-value.risk {
    color: var(--risk);
}

.stButton > button, .stDownloadButton > button {
    border-radius: 4px;
    padding: 0.5rem 1.25rem;
    font-weight: 500;
    font-size: 0.95rem;
    box-shadow: none;
}

.stButton > button {
    background: var(--accent);
    border: 1px solid var(--accent);
}

.stButton > button p {
    color: #FFFFFF;
}

.stButton > button:hover {
    background: #173B5C;
    border-color: #173B5C;
}

.stDownloadButton > button {
    background: var(--surface);
    border: 1px solid var(--accent);
}

.stDownloadButton > button p {
    color: var(--accent);
}

.stDownloadButton > button:hover {
    background: var(--canvas);
}

.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 1px dashed #AEB6C2;
    border-radius: 4px;
    color: var(--ink);
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent);
}

[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: var(--ink);
}

[data-testid="stFileUploaderDropzone"] small {
    color: var(--muted);
}

[data-testid="stFileUploaderDropzone"] svg {
    color: var(--muted);
}

[data-testid="stFileUploaderDropzone"] button {
    background: var(--surface);
    border: 1px solid var(--accent);
    border-radius: 4px;
}

[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button span {
    color: var(--accent);
}

[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileName"] {
    color: var(--ink);
}

[data-testid="stExpander"] details {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 4px;
}

[data-testid="stExpander"] summary p {
    color: var(--ink);
    font-weight: 500;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    border-bottom: 1px solid var(--rule);
}

.stTabs [data-baseweb="tab"] {
    height: 2.75rem;
    padding: 0;
    background: transparent;
}

.stTabs [data-baseweb="tab"] p {
    color: var(--muted);
    font-weight: 500;
}

.stTabs [aria-selected="true"] p {
    color: var(--ink);
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--rule);
    border-radius: 4px;
    background: var(--surface);
}

.status-line {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.95rem;
    margin: 0 0 1rem 0;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1.25rem;
    }
    .app-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .app-title {
        font-size: 1.4rem;
    }
    .strip {
        grid-template-columns: repeat(2, 1fr);
    }
    .strip-cell:nth-child(3) {
        border-left: none;
    }
    .strip-cell:nth-child(n + 3) {
        border-top: 1px solid var(--rule);
    }
}
</style>
"""

for token, value in {
    "__INK__": INK,
    "__MUTED__": MUTED,
    "__RULE__": RULE,
    "__SURFACE__": SURFACE,
    "__CANVAS__": CANVAS,
    "__ACCENT__": ACCENT,
    "__RISK__": RISK,
    "__STABLE__": STABLE,
}.items():
    STYLE = STYLE.replace(token, value)

st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_data(ttl=15, show_spinner=False)
def service_online() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).ok
    except requests.RequestException:
        return False


def call_api(endpoint: str, records: list[dict]) -> dict:
    response = requests.post(
        f"{API_URL}{endpoint}", json=records, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[LABEL_COLUMN], errors="ignore")


def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    payload = feature_frame(df).to_dict(orient="records")
    predictions = pd.DataFrame(call_api("/predict/batch", payload)["predictions"])
    result = df.reset_index(drop=True).copy()
    result.insert(0, "Risk score", predictions["probability"].astype(float).values)
    result.insert(
        0,
        "Decision",
        np.where(predictions["prediction"].astype(int).values == 1, "Flagged", "Cleared"),
    )
    return result


def check_drift(df: pd.DataFrame) -> pd.DataFrame:
    payload = feature_frame(df).to_dict(orient="records")
    report = call_api("/drift", payload)["report"]
    rows = [
        {
            "Feature": feature,
            "Status": "Drift detected" if stats.get("drift_detected") else "Stable",
            "KS statistic": stats.get("ks_statistic"),
            "p-value": stats.get("p_value"),
            "PSI": stats.get("psi"),
        }
        for feature, stats in report.items()
    ]
    table = pd.DataFrame(rows)
    if table["PSI"].isna().all():
        table = table.drop(columns=["PSI"])
    return table.sort_values("KS statistic", ascending=False).reset_index(drop=True)


def risk_bands(scored: pd.DataFrame) -> pd.DataFrame:
    bands = pd.cut(scored["Risk score"], bins=BAND_EDGES, labels=BAND_LABELS)
    grouped = scored.groupby(bands, observed=False)
    table = pd.DataFrame({"Transactions": grouped.size()})
    table["Share"] = table["Transactions"] / max(len(scored), 1)
    if "Amount" in scored.columns:
        table["Amount"] = grouped["Amount"].sum()
    return table.reset_index(names="Risk band")


def risk_histogram(scores: pd.Series) -> alt.Chart:
    counts, edges = np.histogram(scores, bins=20, range=(0.0, 1.0))
    frame = pd.DataFrame(
        {"start": edges[:-1], "end": edges[1:], "Transactions": counts}
    )
    return (
        alt.Chart(frame)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X(
                "start:Q",
                bin="binned",
                title="Risk score",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".1f", labelColor=MUTED, titleColor=MUTED),
            ),
            x2="end:Q",
            y=alt.Y(
                "Transactions:Q",
                title="Transactions (log scale)",
                scale=alt.Scale(type="symlog"),
                axis=alt.Axis(labelColor=MUTED, titleColor=MUTED, gridColor=RULE),
            ),
            tooltip=[
                alt.Tooltip("start:Q", title="From", format=".2f"),
                alt.Tooltip("end:Q", title="To", format=".2f"),
                alt.Tooltip("Transactions:Q", format=","),
            ],
        )
        .properties(height=260)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
    )


def metric_strip(items: list[tuple[str, str, bool]]) -> None:
    cells = "".join(
        f"<div class='strip-cell'><p class='strip-label'>{html.escape(label)}</p>"
        f"<p class='strip-value{' risk' if emphasis else ''}'>{html.escape(value)}</p></div>"
        for label, value, emphasis in items
    )
    st.markdown(f"<div class='strip'>{cells}</div>", unsafe_allow_html=True)


def status_line(label: str, color: str) -> None:
    st.markdown(
        f"<div class='status-line'><span class='dot' style='background:{color}'></span>"
        f"<span>{html.escape(label)}</span></div>",
        unsafe_allow_html=True,
    )


def section(title: str, note: str) -> None:
    st.markdown(
        f"<p class='section-title'>{html.escape(title)}</p>"
        f"<p class='section-note'>{html.escape(note)}</p>",
        unsafe_allow_html=True,
    )


def reset_state_for(signature: str) -> None:
    if st.session_state.get("signature") != signature:
        st.session_state["signature"] = signature
        st.session_state["scored"] = None
        st.session_state["drift"] = None


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    lead = [c for c in ["Decision", "Risk score", "Time", "Amount"] if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    return df[lead + rest]


def style_decisions(df: pd.DataFrame):
    def paint(value: str) -> str:
        color = RISK if value == "Flagged" else STABLE
        return f"color: {color}; font-weight: 600;"

    styler = df.style
    mapper = styler.map if hasattr(styler, "map") else styler.applymap
    return mapper(paint, subset=["Decision"])


def transaction_detail(row: pd.Series) -> pd.DataFrame:
    lead = [c for c in ["Decision", "Risk score", "Time", "Amount"] if c in row.index]
    components = [c for c in row.index if c not in lead and c != LABEL_COLUMN]
    components.sort(key=lambda c: abs(float(row[c])) if pd.notna(row[c]) else 0.0, reverse=True)
    ordered = lead + components

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:,.4f}"
        return str(value)

    return pd.DataFrame(
        {"Field": ordered, "Value": [render(row[c]) for c in ordered]}
    )


online = service_online()
service_color = STABLE if online else RISK
service_text = "Scoring service online" if online else "Scoring service unreachable"

st.markdown(
    "<div class='app-header'><div>"
    "<p class='app-title'>Fraud Monitor</p>"
    "<p class='app-subtitle'>Score card transactions, track input drift against the "
    "training baseline and review which features drive the model.</p></div>"
    f"<div class='service'><span class='dot' style='background:{service_color}'></span>"
    f"<span>{service_text}</span></div></div>",
    unsafe_allow_html=True,
)

section(
    "Transaction file",
    "Upload a CSV with the same columns as the training data. "
    "A Class column, if present, is ignored when scoring.",
)
uploaded_file = st.file_uploader(
    "Transaction file", type=["csv"], label_visibility="collapsed"
)

if uploaded_file is None:
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"The file could not be read as CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The file contains no rows.")
    st.stop()

reset_state_for(f"{uploaded_file.name}:{uploaded_file.size}")

metric_strip(
    [
        ("Transactions", f"{len(data):,}", False),
        (
            "Total amount",
            f"{data['Amount'].sum():,.2f}" if "Amount" in data.columns else "n/a",
            False,
        ),
        ("Columns", f"{data.shape[1]}", False),
        ("Missing values", f"{int(data.isna().sum().sum()):,}", False),
    ]
)

with st.expander("Preview first rows"):
    st.dataframe(data.head(20), use_container_width=True, hide_index=True)

st.write("")
scoring_tab, drift_tab, explain_tab = st.tabs(["Scoring", "Drift", "Explanation"])

with scoring_tab:
    st.write("")
    section(
        "Risk scoring",
        "Each transaction receives a fraud probability. Transactions above the "
        "model decision threshold are flagged for review.",
    )
    if st.button("Score transactions", key="score_button"):
        with st.spinner("Scoring transactions"):
            try:
                st.session_state["scored"] = score_transactions(data)
            except requests.HTTPError as exc:
                st.session_state["scored"] = None
                st.error(
                    f"The scoring service rejected the request "
                    f"(HTTP {exc.response.status_code}). Check that the columns match "
                    f"the training schema."
                )
            except requests.RequestException as exc:
                st.session_state["scored"] = None
                st.error(f"The scoring service at {API_URL} could not be reached: {exc}")

    scored = st.session_state.get("scored")
    if scored is None:
        st.info("Select Score transactions to run the model on this file.")
    else:
        flagged_mask = scored["Decision"] == "Flagged"
        flagged = int(flagged_mask.sum())
        has_amount = "Amount" in scored.columns
        metric_strip(
            [
                ("Scored", f"{len(scored):,}", False),
                ("Flagged", f"{flagged:,}", flagged > 0),
                ("Flag rate", f"{flagged / len(scored):.2%}", False),
                (
                    "Amount flagged",
                    f"{scored.loc[flagged_mask, 'Amount'].sum():,.2f}" if has_amount else "n/a",
                    flagged > 0 and has_amount,
                ),
            ]
        )

        left, right = st.columns([3, 2], gap="large")
        with left:
            section(
                "Score distribution",
                "Legitimate transactions cluster near zero. Mass at the high end is "
                "what the model considers likely fraud.",
            )
            st.altair_chart(risk_histogram(scored["Risk score"]), use_container_width=True)
        with right:
            section("Risk bands", "Transactions grouped by score.")
            band_config = {
                "Transactions": st.column_config.NumberColumn(format="%d"),
                "Share": st.column_config.ProgressColumn(
                    "Share", min_value=0.0, max_value=1.0, format="percent"
                ),
            }
            if has_amount:
                band_config["Amount"] = st.column_config.NumberColumn(format="%.2f")
            st.dataframe(
                risk_bands(scored),
                use_container_width=True,
                hide_index=True,
                column_config=band_config,
            )

        st.write("")
        section(
            "Review queue",
            "Sorted by risk score, highest first. Select a row to inspect the transaction.",
        )
        f1, f2 = st.columns([1, 2], gap="large")
        view = f1.radio("Show", ["Flagged only", "All transactions"], horizontal=True)
        minimum = f2.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.01)

        shown = scored[scored["Risk score"] >= minimum]
        if view == "Flagged only":
            shown = shown[shown["Decision"] == "Flagged"]
        shown = order_columns(shown.sort_values("Risk score", ascending=False))
        visible = shown.head(TABLE_ROW_LIMIT).reset_index(drop=True)

        if visible.empty:
            st.info("No transactions match the current filters.")
        else:
            table_col, detail_col = st.columns([3, 2], gap="large")
            with table_col:
                event = st.dataframe(
                    style_decisions(visible),
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                    on_select="rerun",
                    selection_mode="single-row",
                    column_config={
                        "Risk score": st.column_config.ProgressColumn(
                            "Risk score", min_value=0.0, max_value=1.0, format="%.4f"
                        ),
                    },
                )
                st.caption(f"Showing {len(visible):,} of {len(shown):,} matching rows.")
            with detail_col:
                selected = event.selection.rows if event is not None else []
                if selected:
                    st.markdown("**Transaction detail**")
                    st.dataframe(
                        transaction_detail(visible.iloc[selected[0]]),
                        use_container_width=True,
                        hide_index=True,
                        height=420,
                    )
                else:
                    st.markdown("**Transaction detail**")
                    st.caption("No row selected.")

            st.download_button(
                "Download all results as CSV",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="scored_transactions.csv",
                mime="text/csv",
            )

