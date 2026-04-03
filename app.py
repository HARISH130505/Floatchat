# app.py
# ═════════════════════════════════════════════════════════════════════════════
# FloatChat — ARGO Ocean Analytics System
# ─────────────────────────────────────────────────────────────────────────────
#  Entry point for the Streamlit application.
#
#  Pages (sidebar navigation):
#    1. 💬 Chat         — Hybrid TF-IDF + data chatbot
#    2. 📊 Dashboard    — Plotly charts on ARGO global data
#    3. 🔍 Anomaly      — Anomaly detection (Z-score / Isolation Forest)
#    4. 📈 Forecasting  — Time-series trend and future projection
#
#  Run with:
#    streamlit run app.py
# ═════════════════════════════════════════════════════════════════════════════

from datetime import datetime
from html import escape

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config — MUST be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloatChat — ARGO Ocean Analytics",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Import modules (after page config)
# ─────────────────────────────────────────────────────────────────────────────
from modules.database  import DataManager
from modules.qc_guard  import apply_qc, qc_summary
from modules.chatbot   import FloatChatbot
from modules.dashboard import (
    plot_profiles_over_time,
    plot_ocean_distribution,
    plot_profiler_types,
    plot_top_institutions,
    plot_geo_scatter,
)
from modules.anomaly  import detect_anomalies
from modules.forecast import forecast_trend

# ═════════════════════════════════════════════════════════════════════════════
# Global CSS — Dark ocean theme
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ────────────────────────────────────────────────── */
html, body, [class*="stApp"] {
    font-family: 'Inter', sans-serif !important;
    background: #020d1a !important;
    color: #c9d9e8 !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020d1a 0%, #031628 100%) !important;
    border-right: 1px solid rgba(0, 180, 216, 0.15) !important;
}
[data-testid="stSidebar"] * { color: #c9d9e8 !important; }

/* ── Sidebar logo area ───────────────────────────────────────────── */
.sidebar-logo {
    text-align: center;
    padding: 20px 0 10px;
    border-bottom: 1px solid rgba(0,180,216,0.15);
    margin-bottom: 20px;
}
.sidebar-logo .brand { font-size: 28px; font-weight: 700; color: #00b4d8; letter-spacing: -0.5px; }
.sidebar-logo .sub   { font-size: 12px; color: rgba(144,213,236,0.55); margin-top: 2px; }

/* ── Sidebar nav buttons ─────────────────────────────────────────── */
.stButton > button {
    width: 100%;
    background: rgba(0,25,45, 0.6) !important;
    border: 1px solid rgba(0,180,216,0.3) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 12px 14px !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    margin-bottom: 4px;
}
.stButton > button:hover {
    background: rgba(0,180,216,0.2) !important;
    border-color: rgba(0,180,216,0.6) !important;
    color: #ffffff !important;
    transform: translateX(2px);
}

/* ── Active nav button ───────────────────────────────────────────── */
.nav-active > button {
    background: rgba(0,180,216,0.20) !important;
    border-color: #00b4d8 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* ── Metric cards ────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(0,180,216,0.06);
    border: 1px solid rgba(0,180,216,0.18);
    border-radius: 14px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"]  { color: rgba(144,213,236,0.7) !important; font-size: 13px; }
[data-testid="stMetricValue"]  { color: #00b4d8 !important; font-size: 28px !important; font-weight: 700 !important; }

/* ── Chat bubbles ────────────────────────────────────────────────── */
.chat-container { max-height: 500px; overflow-y: auto; padding: 10px 4px; }
.message { display:flex; align-items:flex-end; margin:12px 0; animation:fadeUp 0.35s ease; }
.message.user { justify-content: flex-end; }
.message.bot  { justify-content: flex-start; }
@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.avatar {
    width:36px;height:36px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:16px;flex-shrink:0;margin:0 8px;
}
.avatar-user { background:linear-gradient(135deg,#0077b6,#00b4d8); box-shadow:0 2px 10px rgba(0,120,180,.35); }
.avatar-bot  { background:linear-gradient(135deg,#023e8a,#0096c7); box-shadow:0 2px 10px rgba(2,62,138,.35); }
.bubble { padding:12px 18px; max-width:74%; font-size:14px; line-height:1.6; word-wrap:break-word; }
.user-bubble {
    background:linear-gradient(135deg,#0077b6,#00b4d8);
    color:#fff; border-radius:20px 20px 4px 20px;
    box-shadow:0 4px 12px rgba(0,180,216,.28);
}
.bot-bubble {
    background:rgba(2,62,138,0.28);
    color:#e0f0ff;
    border:1px solid rgba(0,180,216,0.20);
    border-radius:20px 20px 20px 4px;
    box-shadow:0 4px 12px rgba(0,0,0,.12);
}
.msg-time { font-size:10px; color:rgba(150,200,230,.5); padding:0 10px; margin-top:4px; }
.message.user .msg-time { text-align:right; }
.message.bot  .msg-time { text-align:left; }
.source-badge {
    font-size:10px; display:inline-block;
    padding:2px 8px; border-radius:10px; margin-top:4px;
}
.source-data  { background:rgba(0,255,150,.1); color:#00e090; border:1px solid rgba(0,255,150,.2); }
.source-kb    { background:rgba(0,180,216,.1);  color:#48cae4; border:1px solid rgba(0,180,216,.2); }
.source-fallback { background:rgba(255,165,0,.1); color:#ffaa33; border:1px solid rgba(255,165,0,.2); }

/* Chat scrollbar */
.chat-container::-webkit-scrollbar { width:5px; }
.chat-container::-webkit-scrollbar-track { background:transparent; }
.chat-container::-webkit-scrollbar-thumb { background:rgba(0,180,216,.25); border-radius:3px; }

/* ── Welcome screen ──────────────────────────────────────────────── */
.welcome-screen {
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    min-height:360px;text-align:center;animation:fadeUp 0.5s ease;
}
.welcome-icon  { font-size:60px;margin-bottom:14px; }
.welcome-title { font-size:26px;font-weight:700;color:#e0f0ff;margin-bottom:6px; }
.welcome-sub   { font-size:15px;color:rgba(150,200,230,.6);margin-bottom:20px; }

/* ── Suggestion chips ────────────────────────────────────────────── */
.chip-row { display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:16px; }
.chip {
    background:rgba(0,180,216,.10);border:1px solid rgba(0,180,216,.28);
    color:#90d5ec;padding:7px 16px;border-radius:18px;font-size:13px;
    cursor:default;transition:background .2s;
}
.chip:hover { background:rgba(0,180,216,.22); }

/* ── Section headers ─────────────────────────────────────────────── */
.section-header {
    font-size:22px;font-weight:700;color:#00b4d8;
    margin:8px 0 20px;
    border-bottom:1px solid rgba(0,180,216,.18);
    padding-bottom:10px;
}

/* ── QC badge ────────────────────────────────────────────────────── */
.qc-badge {
    background:rgba(0,255,150,.08);border:1px solid rgba(0,255,150,.2);
    color:#00e090;border-radius:8px;padding:8px 14px;
    font-size:13px;margin-bottom:16px;
}

/* ── Info cards ──────────────────────────────────────────────────── */
.info-card {
    background:rgba(0,180,216,.07);border:1px solid rgba(0,180,216,.18);
    border-radius:14px;padding:18px 22px;margin-bottom:16px;
}

/* ── Streamlit overrides ─────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border:1px solid rgba(0,180,216,.22) !important;
    border-radius:16px !important;
    background:transparent !important;
}
.stTextInput > div > div > input {
    background:rgba(0,180,216,.07) !important;
    border:1px solid rgba(0,180,216,.25) !important;
    border-radius:10px !important;
    color:#e0f0ff !important;
}
.stSelectbox > div > div { border-color:rgba(0,180,216,.25) !important; }

div[data-testid="stTabs"] [data-baseweb="tab"] { color:#90d5ec !important; }
div[data-testid="stTabs"] [aria-selected="true"] { color:#00b4d8 !important; border-bottom-color:#00b4d8 !important; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initialising data manager…")
def get_data_manager() -> DataManager:
    return DataManager()


@st.cache_resource(show_spinner="Loading chatbot…")
def get_chatbot() -> FloatChatbot:
    return FloatChatbot()


@st.cache_data(show_spinner="Loading ARGO data…")
def get_raw_data(_dm: DataManager):
    """Cache the raw DataFrame (QC applied at display time)."""
    return _dm.get_profiles()


def _init_state():
    defaults = {
        "page":     "chat",
        "messages": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# Heavy singletons
dm      = get_data_manager()
chatbot = get_chatbot()
raw_df  = get_raw_data(dm)

# Apply QC filter (cached per raw_df hash is not trivial; apply each render but cheap)
qc_df, qc_status = apply_qc(raw_df)


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="brand">🌊 FloatChat</div>
        <div class="sub">ARGO Ocean Analytics System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")
    nav_items = [
        ("chat",      "💬 Chat",               "Ask questions about ARGO data"),
        ("dashboard", "📊 Dashboard",           "Interactive data visualisations"),
        ("anomaly",   "🔍 Anomaly Detection",   "Detect unusual patterns"),
        ("forecast",  "📈 Forecasting",         "Predict future trends"),
    ]
    for page_id, label, desc in nav_items:
        is_active = st.session_state["page"] == page_id
        # Streamlit doesn't support dynamic button classes, so we toggle via key naming
        if st.button(label, key=f"nav_{page_id}", help=desc):
            st.session_state["page"] = page_id
            st.rerun()

    st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# Page: Chat
# ═════════════════════════════════════════════════════════════════════════════

def _render_messages():
    """Render the full message history as styled HTML bubbles."""
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    if not st.session_state["messages"]:
        st.markdown("""
        <div class='welcome-screen'>
            <div class='welcome-icon'>🌊</div>
            <div class='welcome-title'>How can I help you?</div>
            <div class='welcome-sub'>Ask me anything about ARGO ocean data</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for m in st.session_state["messages"]:
        role   = m["role"]
        text   = escape(m["text"])
        ts     = m.get("time", "")
        source = m.get("source", "")

        badge_class = {
            "data":           "source-data",
            "knowledge_base": "source-kb",
            "fallback":       "source-fallback",
        }.get(source, "")
        badge_label = {
            "data":           "📊 Live Data",
            "knowledge_base": "📚 Knowledge Base",
            "fallback":       "💡 Best Guess",
        }.get(source, "")
        badge_html = f"<div class='source-badge {badge_class}'>{badge_label}</div>" if badge_label else ""

        if role == "user":
            st.markdown(f"""
            <div class='message user'>
                <div>
                    <div class='bubble user-bubble'>{text}</div>
                    <div class='msg-time'>{ts}</div>
                </div>
                <div class='avatar avatar-user'>👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='message bot'>
                <div class='avatar avatar-bot'>🤖</div>
                <div>
                    <div class='bubble bot-bubble'>{text}</div>
                    {badge_html}
                    <div class='msg-time'>{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


SUGGESTIONS = [
    "Tell me about the Pacific Ocean",
    "How many profiles are there?",
    "What is the ARGO project?",
    "Top contributing institutions",
    "When was the latest data?",
    "How do ARGO floats send data?",
]


def _process_query(text: str):
    now = datetime.now().strftime("%H:%M")
    st.session_state["messages"].append({"role": "user", "text": text, "time": now})

    response = chatbot.respond(text, qc_df)

    st.session_state["messages"].append({
        "role":   "bot",
        "text":   response["text"],
        "time":   now,
        "source": response["source"],
    })


if st.session_state["page"] == "chat":
    st.markdown("<div class='section-header'>💬 FloatChat Assistant</div>", unsafe_allow_html=True)

    chat_box = st.container(border=True)
    with chat_box:
        _render_messages()

        # Suggestion chips (only before first message)
        if not st.session_state["messages"]:
            cols = st.columns(3)
            for i, sug in enumerate(SUGGESTIONS):
                with cols[i % 3]:
                    if st.button(sug, key=f"sug_{i}", use_container_width=True):
                        _process_query(sug)
                        st.rerun()

        # Input form
        with st.form("chat_form", clear_on_submit=True):
            c1, c2 = st.columns([8, 1])
            with c1:
                user_text = st.text_input(
                    "Message",
                    label_visibility="collapsed",
                    placeholder="Ask about ARGO floats, oceans, institutions…",
                )
            with c2:
                send = st.form_submit_button("➤", use_container_width=True)

        if send and user_text and user_text.strip():
            _process_query(user_text.strip())
            st.rerun()

    # Clear chat
    if st.session_state["messages"]:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state["messages"] = []
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# Page: Dashboard
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state["page"] == "dashboard":
    st.markdown("<div class='section-header'>📊 ARGO Ocean Dashboard</div>", unsafe_allow_html=True)

    if qc_df.empty:
        st.warning("No data loaded. Ensure argo_all.csv is in the data/ folder.")
    else:
        # ── Metric cards ──────────────────────────────────────────────
        stats = dm.get_summary_stats(qc_df)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📋 Total Profiles",   f"{stats['total_profiles']:,}")
        m2.metric("🌍 Unique Oceans",    stats["unique_oceans"])
        m3.metric("🏢 Institutions",     stats["unique_institutions"])
        m4.metric("📅 Date Range",       f"{stats['date_min'][:4]}–{stats['date_max'][:4]}")

        st.markdown("---")

        # ── Sub-tabs ──────────────────────────────────────────────────
        dash_tab1, dash_tab2 = st.tabs(["📊 Overview", "🔬 Float Explorer"])

        with dash_tab1:
            # ── Chart grid ────────────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_profiles_over_time(qc_df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_ocean_distribution(qc_df), use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(plot_profiler_types(qc_df), use_container_width=True)
            with col4:
                st.plotly_chart(plot_top_institutions(qc_df), use_container_width=True)

            # ── Geo map ───────────────────────────────────────────────
            st.markdown("---")
            with st.spinner("Rendering global map…"):
                st.plotly_chart(plot_geo_scatter(qc_df, sample=15_000), use_container_width=True)

            # ── Raw data preview ──────────────────────────────────────
            with st.expander("🔎 Raw Data Preview (first 500 rows)"):
                st.dataframe(qc_df.head(500), use_container_width=True, hide_index=True)

        with dash_tab2:
            # ── Float Explorer (profiles.csv + metadata.csv merged) ───
            if not dm.has_enriched_data:
                st.info(
                    "Float Explorer requires **profiles.csv** and **metadata.csv** "
                    "in the data/ folder. Run `ingest/fetch_and_prepare.py` to generate them."
                )
            else:
                with st.spinner("Loading enriched float data…"):
                    enriched = dm.get_merged_profiles()

                if enriched.empty:
                    st.warning("Enriched dataset is empty.")
                else:
                    # ── Enriched metrics ──────────────────────────────
                    e1, e2, e3 = st.columns(3)
                    e1.metric("🛰️ Unique Floats",     f"{enriched['file'].nunique():,}"
                              if "file" in enriched.columns else "N/A")
                    e2.metric("📋 Enriched Profiles",  f"{len(enriched):,}")
                    e3.metric("🕒 Last Update",
                              enriched["date_update"].max().strftime("%Y-%m-%d")
                              if "date_update" in enriched.columns
                              and not enriched["date_update"].isna().all()
                              else "N/A")

                    st.markdown("---")

                    # ── Per-float activity table ───────────────────────
                    activity = dm.get_float_activity(enriched)
                    if not activity.empty:
                        st.markdown("##### 🛰️ Per-Float Activity")

                        # Search filter
                        search = st.text_input(
                            "Filter by float file, ocean, or institution…",
                            placeholder="e.g. atlantic, AOML, 5900142",
                            key="float_search",
                        )
                        if search:
                            mask = activity.apply(
                                lambda row: search.lower() in str(row.values).lower(), axis=1
                            )
                            activity = activity[mask]

                        st.dataframe(
                            activity.head(500),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "file":          st.column_config.TextColumn("Float File"),
                                "profile_count": st.column_config.NumberColumn("Profiles", format="%d"),
                                "first_seen":    st.column_config.DatetimeColumn("First Seen",  format="YYYY-MM-DD"),
                                "last_seen":     st.column_config.DatetimeColumn("Last Seen",   format="YYYY-MM-DD"),
                                "last_update":   st.column_config.DatetimeColumn("Last Update", format="YYYY-MM-DD"),
                            },
                        )

                    # ── Last-update distribution ───────────────────────
                    if "date_update" in enriched.columns:
                        import plotly.graph_objects as go
                        upd = enriched.dropna(subset=["date_update"])
                        if not upd.empty:
                            ts = upd.groupby(upd["date_update"].dt.to_period("M")).size()
                            ts.index = ts.index.to_timestamp()
                            fig = go.Figure(go.Bar(
                                x=ts.index, y=ts.values,
                                marker_color="#00b4d8",
                                hovertemplate="<b>%{x|%b %Y}</b><br>Updates: %{y:,}<extra></extra>",
                            ))
                            fig.update_layout(
                                title="🕒 Float Last-Update Activity",
                                xaxis_title="Month", yaxis_title="Floats Updated",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(10,20,40,0.5)",
                                font=dict(family="Inter, sans-serif", color="#c9d9e8"),
                                margin=dict(l=10, r=10, t=40, b=10),
                            )
                            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Page: Anomaly Detection
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state["page"] == "anomaly":
    st.markdown("<div class='section-header'>🔍 Anomaly Detection</div>", unsafe_allow_html=True)

    if qc_df.empty:
        st.warning("No data loaded.")
    else:
        # ── Controls ──────────────────────────────────────────────────
        with st.container(border=True):
            st.markdown("**Detection Settings**")
            c1, c2, c3 = st.columns(3)
            with c1:
                method = st.selectbox(
                    "Algorithm",
                    ["zscore", "isolation_forest"],
                    format_func=lambda x: "Z-Score" if x == "zscore" else "Isolation Forest",
                    help="Z-Score is fast; Isolation Forest handles multi-dimensional anomalies.",
                )
            with c2:
                target = st.selectbox(
                    "Target",
                    ["count", "spatial"],
                    format_func=lambda x: "Monthly Profile Count" if x == "count" else "Spatial (lat/lon)",
                )
            with c3:
                if method == "zscore":
                    z_thresh = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.5)
                    contamination = 0.05
                else:
                    contamination = st.slider("Contamination (%)", 1, 20, 5, 1) / 100
                    z_thresh = 3.0

            run_btn = st.button("🚀 Run Detection", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Running anomaly detection…"):
                result = detect_anomalies(
                    qc_df,
                    method=method,
                    target=target,
                    zscore_threshold=z_thresh,
                    contamination=contamination,
                )

            # ── Results ───────────────────────────────────────────────
            a1, a2, a3 = st.columns(3)
            a1.metric("🔎 Anomalies Found",  result["anomaly_count"])
            a2.metric("📊 Total Analysed",   result["total"])
            a3.metric("🧮 Method",            result["method"])

            st.markdown(f"""
            <div class='info-card'>{result['summary']}</div>
            """, unsafe_allow_html=True)

            st.plotly_chart(result["figure"], use_container_width=True)

            if not result["flagged_df"].empty:
                with st.expander(f"📋 Flagged Records ({result['anomaly_count']})"):
                    st.dataframe(result["flagged_df"], use_container_width=True, hide_index=True)
        else:
            st.info("Configure settings above and click **Run Detection** to analyse the dataset.")

            # ── Method explainers ────────────────────────────────────
            e1, e2 = st.columns(2)
            with e1:
                st.markdown("""
                <div class='info-card'>
                    <strong>📐 Z-Score</strong><br><br>
                    Flags data points where the standardised value deviates
                    more than <em>N</em> standard deviations from the mean.<br><br>
                    ✅ Fast &nbsp;|&nbsp; ✅ Interpretable &nbsp;|&nbsp; ⚠️ Assumes normality
                </div>
                """, unsafe_allow_html=True)
            with e2:
                st.markdown("""
                <div class='info-card'>
                    <strong>🌲 Isolation Forest</strong><br><br>
                    An ensemble tree method that isolates anomalies by randomly
                    partitioning the feature space. Works well on multi-dimensional data.<br><br>
                    ✅ Multi-dimensional &nbsp;|&nbsp; ✅ No distributional assumptions
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Page: Forecasting
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state["page"] == "forecast":
    st.markdown("<div class='section-header'>📈 Profile Count Forecasting</div>", unsafe_allow_html=True)

    if qc_df.empty:
        st.warning("No data loaded.")
    else:
        # ── Controls ──────────────────────────────────────────────────
        with st.container(border=True):
            st.markdown("**Forecast Settings**")
            c1, c2, c3 = st.columns(3)
            with c1:
                f_method = st.selectbox(
                    "Method",
                    ["linear", "arima"],
                    format_func=lambda x: "Linear Regression" if x == "linear" else "ARIMA",
                    help="Linear Regression is always available. ARIMA requires statsmodels.",
                )
            with c2:
                f_periods = st.slider("Forecast Horizon (months)", 3, 36, 12)
            with c3:
                if f_method == "arima":
                    p = st.number_input("ARIMA p (AR order)",  0, 5, 1)
                    d = st.number_input("ARIMA d (Diff order)", 0, 2, 1)
                    q = st.number_input("ARIMA q (MA order)",  0, 5, 1)
                    arima_order = (int(p), int(d), int(q))
                else:
                    arima_order = (1, 1, 1)
                    st.markdown("""
                    <div style='color:rgba(144,213,236,.5);font-size:12px;margin-top:16px;'>
                        Linear regression fits a trend line to historical monthly counts.
                    </div>
                    """, unsafe_allow_html=True)

            run_f = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

        if run_f:
            with st.spinner("Fitting model and generating forecast…"):
                result = forecast_trend(
                    qc_df,
                    periods=f_periods,
                    method=f_method,
                    arima_order=arima_order if f_method == "arima" else (1, 1, 1),
                )

            st.markdown(f"""
            <div class='info-card'>{result['summary']}</div>
            """, unsafe_allow_html=True)

            st.plotly_chart(result["figure"], use_container_width=True)

            if not result["forecast_df"].empty:
                with st.expander("📋 Forecast Table"):
                    st.dataframe(result["forecast_df"], use_container_width=True, hide_index=True)
        else:
            st.info("Configure settings above and click **Generate Forecast** to run the model.")

            # ── Method explainers ────────────────────────────────────
            e1, e2 = st.columns(2)
            with e1:
                st.markdown("""
                <div class='info-card'>
                    <strong>📏 Linear Regression</strong><br><br>
                    Fits a straight trend line to the historical monthly profile count.
                    Simple, fast, and always works — ideal when the trend is roughly linear.<br><br>
                    ✅ Always available &nbsp;|&nbsp; ✅ Interpretable slope
                </div>
                """, unsafe_allow_html=True)
            with e2:
                st.markdown("""
                <div class='info-card'>
                    <strong>📡 ARIMA</strong><br><br>
                    AutoRegressive Integrated Moving Average — accounts for autocorrelation
                    and local structure in the time series. More accurate for non-linear trends.<br><br>
                    ✅ Handles seasonality &nbsp;|&nbsp; ✅ Confidence intervals &nbsp;|&nbsp; ⚠️ Requires statsmodels
                </div>
                """, unsafe_allow_html=True)