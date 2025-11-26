from datetime import datetime
from html import escape
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FloatChat — ARGO Ocean Assistant", layout="wide", page_icon="🌊")

# ──────────────────────────────────────
# Load Knowledge Base
# ──────────────────────────────────────
def load_kb_from_csv(path="data/knowledge_base.csv"):
    try:
        df = pd.read_csv(path)
        if "question" in df.columns and "answer" in df.columns:
            return df[["question","answer"]].dropna().reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame([
        {"question":"What is the ARGO project?", "answer":"ARGO is an AI-powered ocean data system."},
        {"question":"How to run locally?", "answer":"Create venv, install requirements, run: streamlit run app.py"}
    ])

# ──────────────────────────────────────
# Load ARGO data
# ──────────────────────────────────────
OCEAN_NAMES = {"P": "Pacific Ocean", "A": "Atlantic Ocean", "I": "Indian Ocean"}

def load_usage(path="data/argo_all.csv"):
    try:
        df = pd.read_csv(path)
        df = df.rename(columns={"date":"timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        for col in ['profiler_type','ocean','institution']:
            if col in df.columns:
                df[col] = df[col].astype(str)
        if 'ocean' in df.columns:
            df['ocean'] = df['ocean'].map(OCEAN_NAMES).fillna(df['ocean'])
        return df
    except Exception:
        return pd.DataFrame()

# ──────────────────────────────────────
# Retrieval Chatbot
# ──────────────────────────────────────
def build_vectorizer(questions_list):
    if not questions_list:
        return None, None
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    X = vectorizer.fit_transform(questions_list)
    return vectorizer, X

def answer_query(query, vectorizer, X, kb_df, threshold=0.35):
    if vectorizer is None or X is None or kb_df.shape[0] == 0:
        return ("Knowledge base is empty.", 0, None)
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score >= threshold:
        return kb_df.iloc[best_idx]['answer'], best_score, kb_df.iloc[best_idx]['question']
    if best_score > 0:
        suggestion = kb_df.iloc[best_idx]['question']
        return f"I'm not entirely sure about that, but here's the closest topic I know: \"{suggestion}\" (confidence: {best_score:.0%}).", best_score, suggestion
    return "I don't have information on that yet. Try asking about ARGO floats, oceans, or profiler types!", best_score, None

# ──────────────────────────────────────
# Natural Language Response Formatting
# ──────────────────────────────────────
def _top_items(d, n=3):
    """Return the top-n items from a dict as a readable sentence fragment."""
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]
    parts = []
    for name, count in sorted_items:
        clean = str(name).strip()
        if clean in ("nan", ""):
            clean = "Unknown"
        parts.append(f"{clean} ({count:,})")
    return ", ".join(parts)

def format_data_response(num_profiles, oceans, prof_types, institutions):
    """Build a conversational response from data aggregation dicts."""
    lines = [f"Based on the ARGO dataset, there are **{num_profiles:,} profiles** on record."]
    if oceans:
        lines.append(f"The most represented oceans are: {_top_items(oceans)}.")
    if prof_types:
        lines.append(f"The leading profiler types are: {_top_items(prof_types)}.")
    if institutions:
        lines.append(f"Top contributing institutions include: {_top_items(institutions, 5)}.")
    return " ".join(lines)

# ──────────────────────────────────────
# ARGO Query Router
# ──────────────────────────────────────
SPECIFIC_OCEANS = {
    'pacific':  'Pacific Ocean',
    'atlantic': 'Atlantic Ocean',
    'indian':   'Indian Ocean',
}

def data_query(query, df):
    """Parse the user's question and return only the relevant data slice."""
    if df.empty:
        return "No ARGO data is loaded at the moment."
    q = query.lower()

    # Check if user is asking about a specific ocean
    for keyword, ocean_name in SPECIFIC_OCEANS.items():
        if keyword in q and 'ocean' in df.columns:
            subset = df[df['ocean'] == ocean_name]
            count = len(subset)
            if count == 0:
                return f"There are no profiles recorded for the {ocean_name} in this dataset."
            parts = [f"The {ocean_name} has {count:,} profiles in the ARGO dataset."]
            if 'profiler_type' in df.columns:
                pt = subset['profiler_type'].value_counts()
                parts.append(f"Profiler types used: {_top_items(pt.to_dict(), 3)}.")
            if 'institution' in df.columns:
                inst = subset['institution'].value_counts()
                parts.append(f"Contributing institutions: {_top_items(inst.to_dict(), 5)}.")
            return " ".join(parts)

    num_profiles = len(df)

    # Determine what the user is asking about
    wants_ocean = any(w in q for w in ['ocean', 'sea'])
    wants_profiler = any(w in q for w in ['profiler', 'sensor', 'float type', 'type'])
    wants_institution = any(w in q for w in ['institution', 'organization', 'who', 'contributor'])
    wants_count = any(w in q for w in ['how many', 'count', 'total', 'number of'])

    # If nothing specific matched, give a brief overview
    if not any([wants_ocean, wants_profiler, wants_institution, wants_count]):
        wants_ocean = wants_profiler = wants_institution = True

    parts = [f"The ARGO dataset contains {num_profiles:,} profiles in total."]

    if wants_ocean and 'ocean' in df.columns:
        oc = df['ocean'].value_counts()
        top = _top_items(oc.to_dict(), 5)
        parts.append(f"Ocean distribution: {top}.")

    if wants_profiler and 'profiler_type' in df.columns:
        pt = df['profiler_type'].value_counts()
        top = _top_items(pt.to_dict(), 3)
        parts.append(f"Profiler types: {top}.")

    if wants_institution and 'institution' in df.columns:
        inst = df['institution'].value_counts()
        top = _top_items(inst.to_dict(), 5)
        parts.append(f"Top institutions: {top}.")

    if wants_count and not any([wants_ocean, wants_profiler, wants_institution]):
        parts = [f"There are {num_profiles:,} profiles in the ARGO dataset."]

    return " ".join(parts)

# ──────────────────────────────────────
# Session State
# ──────────────────────────────────────
if 'kb_df' not in st.session_state:
    st.session_state['kb_df'] = load_kb_from_csv()
    st.session_state['vectorizer'], st.session_state['kb_X'] = build_vectorizer(
        st.session_state['kb_df']['question'].tolist()
    )
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'usage' not in st.session_state:
    st.session_state['usage'] = load_usage()
if 'similarity_threshold' not in st.session_state:
    st.session_state['similarity_threshold'] = 0.35

# ──────────────────────────────────────
# Premium Chat CSS
# ──────────────────────────────────────
CHAT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ---------- globals ---------- */
html, body, [class*="stApp"] {
    font-family: 'Inter', sans-serif !important;
}

/* ---------- chat wrapper (ocean-bordered box) ---------- */
.chat-wrapper {
    border: 2px solid rgba(0, 180, 216, 0.45);
    border-radius: 16px;
    padding: 6px;
    background: transparent;
    box-shadow: 0 0 20px rgba(0, 119, 182, 0.10),
                inset 0 0 30px rgba(0, 180, 216, 0.03);
}

/* ---------- chat container (scrollable inner) ---------- */
.chat-container {
    max-height: 520px;
    overflow-y: auto;
    padding: 18px 14px;
    background: transparent;
}

/* ---------- message row ---------- */
.message {
    display: flex;
    align-items: flex-end;
    margin: 10px 0;
    animation: fadeSlideIn 0.35s ease;
}
.message.user { justify-content: flex-end; }
.message.bot  { justify-content: flex-start; }

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---------- avatars ---------- */
.avatar {
    width: 38px; height: 38px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    margin: 0 10px;
    flex-shrink: 0;
}
.avatar-user {
    background: linear-gradient(135deg, #0077b6, #00b4d8);
    color: #fff;
    box-shadow: 0 2px 10px rgba(0, 119, 182, 0.35);
}
.avatar-bot {
    background: linear-gradient(135deg, #023e8a, #0096c7);
    color: #fff;
    box-shadow: 0 2px 10px rgba(2, 62, 138, 0.35);
}

/* ---------- bubbles ---------- */
.bubble {
    padding: 14px 20px;
    max-width: 72%;
    font-size: 14.5px;
    line-height: 1.55;
    word-wrap: break-word;
}
.user-bubble {
    background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
    color: #fff;
    border-radius: 22px 22px 4px 22px;
    box-shadow: 0 4px 14px rgba(0, 180, 216, 0.30);
}
.bot-bubble {
    background: rgba(2, 62, 138, 0.25);
    color: #e0f0ff;
    border: 1px solid rgba(0, 180, 216, 0.20);
    border-radius: 22px 22px 22px 4px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
}

/* ---------- timestamp ---------- */
.msg-time {
    font-size: 10px;
    color: rgba(150, 200, 230, 0.6);
    margin-top: 4px;
    padding: 0 12px;
}
.message.user .msg-time { text-align: right; }
.message.bot  .msg-time { text-align: left; }

/* ---------- scrollbar ---------- */
.chat-container::-webkit-scrollbar { width: 6px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb {
    background: rgba(0, 180, 216, 0.25);
    border-radius: 3px;
}
.chat-container::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 180, 216, 0.45);
}

/* ---------- override streamlit bordered container ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 2px solid rgba(0, 180, 216, 0.40) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 24px rgba(0, 119, 182, 0.12),
                inset 0 0 30px rgba(0, 180, 216, 0.03) !important;
    background: transparent !important;
    padding: 8px !important;
}

/* ---------- welcome screen ---------- */
.welcome-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 340px;
    text-align: center;
    animation: fadeSlideIn 0.5s ease;
}
.welcome-icon {
    font-size: 56px;
    margin-bottom: 12px;
}
.welcome-title {
    font-size: 26px;
    font-weight: 600;
    color: #e0f0ff;
    margin-bottom: 6px;
}
.welcome-sub {
    font-size: 15px;
    color: rgba(150, 200, 230, 0.6);
    margin-bottom: 24px;
}
.suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
}
.suggestion-chip {
    background: rgba(0, 180, 216, 0.12);
    border: 1px solid rgba(0, 180, 216, 0.30);
    color: #90d5ec;
    padding: 8px 18px;
    border-radius: 20px;
    font-size: 13px;
    cursor: default;
    transition: background 0.2s;
}
.suggestion-chip:hover {
    background: rgba(0, 180, 216, 0.22);
}
</style>
"""

st.markdown(CHAT_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────
# Render Chat Messages
# ──────────────────────────────────────
def render_chat_messages():
    if not st.session_state['messages']:
        st.markdown("""
        <div class='welcome-screen'>
            <div class='welcome-icon'>🌊</div>
            <div class='welcome-title'>How can I help you today?</div>
            <div class='welcome-sub'>Ask me anything about ARGO ocean data</div>
        </div>
        """, unsafe_allow_html=True)
        return
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for m in st.session_state['messages']:
        role = m.get('role')
        text = escape(m.get('text', ''))
        time_str = m.get('time', '')
        if role == 'user':
            st.markdown(f"""
            <div class='message user'>
                <div>
                    <div class='bubble user-bubble'>{text}</div>
                    <div class='msg-time'>{time_str}</div>
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
                    <div class='msg-time'>{time_str}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────
# Layout
# ──────────────────────────────────────
st.title("🌊 FloatChat — ARGO Ocean Assistant")
tab1, tab2 = st.tabs(["💬 Chatbot", "📊 Dashboard"])

# ════════════════ CHATBOT TAB ════════════════
SUGGESTIONS = [
    "Tell me about Pacific Ocean",
    "How many profiles are there?",
    "What is the ARGO project?",
    "Top contributing institutions",
]

def _process_query(text):
    """Process a user query and append both user + bot messages."""
    now = datetime.now().strftime("%H:%M")
    st.session_state['messages'].append({'role':'user', 'text':text, 'time':now})
    q_lower = text.lower()
    if any(k in q_lower for k in ['profiles','ocean','institution','profiler','how many','count','total','float','pacific','atlantic','indian']):
        ans = data_query(text, st.session_state['usage'])
    else:
        ans, score, _ = answer_query(
            text,
            st.session_state['vectorizer'],
            st.session_state['kb_X'],
            st.session_state['kb_df'],
            st.session_state['similarity_threshold']
        )
    st.session_state['messages'].append({'role':'bot', 'text':ans, 'time':now})

with tab1:
    chat_box = st.container(border=True)
    with chat_box:
        render_chat_messages()
        # Show suggestion buttons when no messages yet
        if not st.session_state['messages']:
            chip_cols = st.columns(len(SUGGESTIONS))
            for i, suggestion in enumerate(SUGGESTIONS):
                with chip_cols[i]:
                    if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                        _process_query(suggestion)
                        st.rerun()
        with st.form(key="chat_form", clear_on_submit=True):
            cols = st.columns([6, 1])
            with cols[0]:
                user_text = st.text_input(
                    "Message",
                    label_visibility="collapsed",
                    placeholder="Ask me about ARGO floats, oceans, institutions…"
                )
            with cols[1]:
                send = st.form_submit_button("➤", use_container_width=True)
        if send and user_text and user_text.strip():
            _process_query(user_text.strip())
            st.rerun()

# ════════════════ DASHBOARD TAB ════════════════
with tab2:
    usage = st.session_state['usage']

    if usage.empty:
        st.warning("No ARGO data loaded. Place argo_all.csv in the data/ folder.")
    else:
        st.subheader("🌊 ARGO Ocean Data Dashboard")

        # ── Metric Cards ──
        total_profiles = len(usage)
        unique_oceans = usage['ocean'].nunique() if 'ocean' in usage.columns else 0
        unique_inst   = usage['institution'].nunique() if 'institution' in usage.columns else 0
        date_min = usage['timestamp'].min().strftime("%Y-%m-%d") if 'timestamp' in usage.columns else "N/A"
        date_max = usage['timestamp'].max().strftime("%Y-%m-%d") if 'timestamp' in usage.columns else "N/A"

        mc = st.columns(4)
        mc[0].metric("📋 Total Profiles", f"{total_profiles:,}")
        mc[1].metric("🌍 Unique Oceans", unique_oceans)
        mc[2].metric("🏢 Institutions", unique_inst)
        mc[3].metric("📅 Date Range", f"{date_min[:4]}–{date_max[:4]}")

        st.markdown("---")

        # ── Chart Grid (2×2) ──
        row1_c1, row1_c2 = st.columns(2)
        row2_c1, row2_c2 = st.columns(2)

        # ---- 1. Profiles over time ----
        with row1_c1:
            st.markdown("##### 📈 Profiles Over Time")
            ts = usage.groupby(usage['timestamp'].dt.to_period('M')).size()
            ts.index = ts.index.to_timestamp()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.fill_between(ts.index, ts.values, alpha=0.25, color='#2575fc')
            ax.plot(ts.index, ts.values, color='#2575fc', linewidth=1.5)
            ax.set_xlabel("Date", fontsize=9)
            ax.set_ylabel("Profiles", fontsize=9)
            ax.tick_params(labelsize=8)
            fig.autofmt_xdate(rotation=30)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ---- 2. Ocean Distribution ----
        with row1_c2:
            st.markdown("##### 🌍 Ocean Distribution")
            if 'ocean' in usage.columns:
                ocean_counts = usage['ocean'].value_counts().head(8)
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                colors_ocean = plt.cm.cool(np.linspace(0.2, 0.8, len(ocean_counts)))
                bars = ax2.barh(ocean_counts.index[::-1], ocean_counts.values[::-1], color=colors_ocean[::-1], edgecolor='white', linewidth=0.5)
                ax2.set_xlabel("Number of Profiles", fontsize=9)
                ax2.tick_params(labelsize=8)
                for bar, val in zip(bars, ocean_counts.values[::-1]):
                    ax2.text(bar.get_width() + max(ocean_counts.values)*0.01, bar.get_y() + bar.get_height()/2,
                             f'{val:,}', va='center', fontsize=7, color='#333')
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

        # ---- 3. Profiler Type Distribution ----
        with row2_c1:
            st.markdown("##### ⚙️ Profiler Type Distribution")
            if 'profiler_type' in usage.columns:
                pt_counts = usage['profiler_type'].value_counts().head(8)
                fig3, ax3 = plt.subplots(figsize=(5, 3))
                colors_pt = plt.cm.viridis(np.linspace(0.2, 0.85, len(pt_counts)))
                wedges, texts, autotexts = ax3.pie(
                    pt_counts.values, labels=pt_counts.index, autopct='%1.1f%%',
                    colors=colors_pt, startangle=140, pctdistance=0.78,
                    textprops={'fontsize': 8}
                )
                for t in autotexts:
                    t.set_fontsize(7)
                ax3.set_aspect('equal')
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)

        # ---- 4. Top Institutions ----
        with row2_c2:
            st.markdown("##### 🏢 Top Contributing Institutions")
            if 'institution' in usage.columns:
                inst_counts = usage['institution'].value_counts().head(10)
                fig4, ax4 = plt.subplots(figsize=(5, 3))
                colors_inst = plt.cm.plasma(np.linspace(0.2, 0.85, len(inst_counts)))
                bars4 = ax4.bar(inst_counts.index, inst_counts.values, color=colors_inst, edgecolor='white', linewidth=0.5)
                ax4.set_ylabel("Profiles", fontsize=9)
                ax4.set_xlabel("Institution", fontsize=9)
                ax4.tick_params(labelsize=7, axis='x', rotation=45)
                ax4.tick_params(labelsize=8, axis='y')
                for bar, val in zip(bars4, inst_counts.values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(inst_counts.values)*0.01,
                             f'{val:,}', ha='center', va='bottom', fontsize=6, color='#333')
                fig4.tight_layout()
                st.pyplot(fig4)
                plt.close(fig4)
