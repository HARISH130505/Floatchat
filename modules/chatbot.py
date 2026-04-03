# modules/chatbot.py
# ─────────────────────────────────────────────────────────────────────────────
# FloatChatbot — Hybrid retrieval chatbot
#
# Routing priority (most specific → least):
#   1. STRONG DATA INTENT: query contains explicit data-stat keywords
#      AND no strong KB match found → data engine
#   2. KB MATCH: TF-IDF cosine similarity ≥ threshold → KB answer
#   3. WEAK DATA INTENT: lower-confidence data keywords → data engine
#   4. FALLBACK: generic "I don't know" reply
#
# This ordering ensures knowledge-base questions (e.g. "Who manages ARGO?",
# "What sensors do floats have?") are answered from the KB, not by
# mis-routing to the data engine.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_KB_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Keyword definitions
#
# STRONG_DATA_KEYWORDS  → unambiguously asking for live numbers from the dataset.
#                         These will bypass the KB check.
# WEAK_DATA_KEYWORDS    → might be data-related, but first check the KB.
# KB_PRIORITY_PATTERNS  → regex patterns that strongly suggest the user wants
#                         a factual/explanatory answer, not a data lookup.
#                         Matching any of these prevents data-engine routing.
# ─────────────────────────────────────────────────────────────────────────────

# Phrases that almost always mean "give me a stat from the dataset"
_STRONG_DATA: list[str] = [
    "how many", "total number", "count of", "number of profiles",
    "show me data", "give me stats", "breakdown of", "distribution of",
    "profiles in", "profiles per", "top institutions", "top countries",
]

# Narrowly scoped ocean / geography terms that need live data
_OCEAN_TERMS: list[str] = [
    "pacific ocean", "atlantic ocean", "indian ocean",
    "profiles in pacific", "profiles in atlantic", "profiles in indian",
]

# These patterns signal the user wants an explanation, not a data stat.
# Even if other data-keywords appear in their sentence, route to KB first.
_KB_PRIORITY_PATTERNS: list[str] = [
    r"\bwhat is\b", r"\bwhat are\b", r"\bwhat does\b",
    r"\bwho (is|are|manages|runs|created|invented|funds)\b",
    r"\bhow do(es)?\b", r"\bhow (are|is|was)\b",
    r"\bwhy (do|does|is|are)\b",
    r"\bexplain\b", r"\bdescribe\b", r"\btell me about\b",
    r"\bwhat kind\b", r"\bwhat type\b", r"\bwhat sort\b",
    r"\bdefine\b", r"\bmeaning of\b",
    r"\bpurpose of\b", r"\bgoal of\b",
    r"\bcan i\b", r"\bhow (can|do) i\b",
    r"\brun (this|the)\b", r"\binstall\b",
]

# Compiled regex for performance
_KB_PRIORITY_RE = re.compile("|".join(_KB_PRIORITY_PATTERNS), re.IGNORECASE)


def _is_strong_data_query(q: str) -> bool:
    """True when the query is clearly asking for live dataset statistics."""
    ql = q.lower()
    return any(kw in ql for kw in _STRONG_DATA + _OCEAN_TERMS)


def _has_kb_priority(q: str) -> bool:
    """True when the query looks like an explanatory/factual question.

    If this returns True, we try the KB first before any data routing.
    """
    return bool(_KB_PRIORITY_RE.search(q))


# ─────────────────────────────────────────────────────────────────────────────
# Data engine — live stats from the ARGO DataFrame
# ─────────────────────────────────────────────────────────────────────────────

# Intent keywords for the data engine (only consulted after routing decision)
_DATA_INTENTS: dict[str, list[str]] = {
    "ocean":       ["ocean", "sea", "pacific", "atlantic", "indian"],
    "institution": ["institution", "organization", "contributor"],
    "profiler":    ["profiler", "float type"],
    "date":        ["date", "latest", "oldest", "range", "recent"],
    "location":    ["latitude", "longitude", "location", "region", "coordinate"],
    "count":       ["how many", "count", "total", "number"],
}

_SPECIFIC_OCEANS = {
    "pacific":  "Pacific Ocean",
    "atlantic": "Atlantic Ocean",
    "indian":   "Indian Ocean",
}

# ARGO institution code → full organisation name
_INSTITUTION_NAMES: dict[str, str] = {
    "AO": "AOML / NOAA (USA)",
    "BO": "BODC (UK)",
    "CS": "CSIRO (Australia)",
    "HZ": "CSHOR / CSIRO (Australia)",
    "IF": "IFREMER (France)",
    "IN": "INCOIS (India)",
    "JA": "JAMSTEC (Japan)",
    "KO": "KORDI (South Korea)",
    "ME": "MEDS / DFO (Canada)",
    "NM": "NMDIS (China)",
    "RU": "SIO RAS (Russia)",
    "SP": "SOCIB (Spain)",
    "GE": "BSH (Germany)",
    "CN": "SOA / FIO (China)",
    "IR": "IOPAN (Poland)",
    "MO": "MBARI (USA)",
    "UW": "UW / APL (USA)",
    "SH": "Shanghai Ocean University",
    "PM": "PMEL / NOAA (USA)",
    "CI": "CalOcean (USA)",
}


def _resolve_institution(code: str) -> str:
    """Return the full institution name for a given code, or the raw code if unknown."""
    code = str(code).strip()
    return _INSTITUTION_NAMES.get(code, code)


def _top_n(series: pd.Series, n: int = 5, resolve_institutions: bool = False) -> str:
    """Format a value_counts Series as a readable sentence fragment.

    Filters out NaN / empty values and optionally resolves institution codes.
    """
    # Drop NaN, 'nan' strings, and empty strings
    clean = series[
        series.index.map(lambda x: str(x).strip().lower() not in ("nan", "", "none", "unknown"))
    ].head(n)

    parts = []
    for k, v in clean.items():
        label = _resolve_institution(str(k)) if resolve_institutions else str(k).strip()
        parts.append(f"{label} ({v:,})")
    return ", ".join(parts) if parts else "No data available"


class _DataEngine:
    """Compute natural-language answers from the live ARGO DataFrame."""

    def respond(self, query: str, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "No ARGO data is currently loaded. Please check the data source."

        q = query.lower()
        n = len(df)

        # ── Specific ocean deep-dive ──────────────────────────────────
        for keyword, ocean_name in _SPECIFIC_OCEANS.items():
            if keyword in q and "ocean" in df.columns:
                sub = df[df["ocean"] == ocean_name]
                if sub.empty:
                    return f"No profiles found for the {ocean_name} in the current dataset."
                parts = [f"The **{ocean_name}** has **{len(sub):,}** profiles in the ARGO dataset."]
                if "profiler_type" in df.columns:
                    parts.append(f"Profiler types: {_top_n(sub['profiler_type'].value_counts(), 3)}.")
                if "institution" in df.columns:
                    parts.append(f"Top institutions: {_top_n(sub['institution'].value_counts(), 5, resolve_institutions=True)}.")
                return " ".join(parts)

        # ── Determine which aspects the user wants ────────────────────
        wants = {tag: any(kw in q for kw in kws) for tag, kws in _DATA_INTENTS.items()}

        # Pure count question — answer directly, no extra info
        if wants["count"] and not any(
            wants[k] for k in ["ocean", "profiler", "institution", "date", "location"]
        ):
            return f"There are **{n:,} profiles** in the current ARGO dataset."

        # No specific intent matched → give a broad overview
        is_overview = not any(wants.values())
        if is_overview:
            wants = {k: True for k in wants}

        parts = []

        # Only prepend total count for overviews or when count is explicitly wanted
        if is_overview or wants["count"]:
            parts.append(f"The ARGO dataset contains **{n:,} profiles** in total.")

        if wants["ocean"] and "ocean" in df.columns:
            parts.append(f"Ocean distribution: {_top_n(df['ocean'].value_counts(), 5)}.")

        if wants["profiler"] and "profiler_type" in df.columns:
            parts.append(f"Profiler types: {_top_n(df['profiler_type'].value_counts(), 3)}.")

        if wants["institution"] and "institution" in df.columns:
            parts.append(f"Top institutions: {_top_n(df['institution'].value_counts(), 5, resolve_institutions=True)}.") 
        if wants["date"] and "timestamp" in df.columns:
            d_min = df["timestamp"].min().strftime("%Y-%m-%d")
            d_max = df["timestamp"].max().strftime("%Y-%m-%d")
            parts.append(f"Data spans from **{d_min}** to **{d_max}**.")

        if wants["location"] and {"latitude", "longitude"}.issubset(df.columns):
            lat_m = df["latitude"].mean()
            lon_m = df["longitude"].mean()
            parts.append(f"Mean float location: lat {lat_m:.2f}°, lon {lon_m:.2f}°.")

        return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main chatbot class
# ─────────────────────────────────────────────────────────────────────────────

class FloatChatbot:
    """Hybrid TF-IDF + data-routing chatbot.

    Usage:
        bot = FloatChatbot()
        response = bot.respond("How many profiles are in the Pacific?", df)
    """

    def __init__(self, threshold: float = 0.28):
        # Slightly lower threshold so we catch near-match KB questions better
        self.threshold = threshold
        self._data_engine = _DataEngine()
        self._kb_df, self._vectorizer, self._kb_X = self._load_kb()

    # ── Knowledge-base setup ─────────────────────────────────────────

    def _load_kb(self) -> tuple:
        """Load knowledge base from Database/CSV and build TF-IDF index."""
        from modules.database import DataManager

        fallback = pd.DataFrame([
            {"question": "What is ARGO?",
             "answer": "ARGO is a global ocean observation programme using autonomous profiling floats."},
            {"question": "How do I run the app?",
             "answer": "Create a virtual env, install requirements, then run: streamlit run app.py"},
        ])
        
        try:
            # DataManager automatically routes to Postgres if active, else CSV
            df = DataManager().get_knowledge_base()
            if not df.empty and {"question", "answer"}.issubset(df.columns):
                df = df[["question", "answer"]].dropna().reset_index(drop=True)
            else:
                df = fallback
        except Exception as e:
            logger.warning(f"FloatChatbot failed to load KB: {e}")
            df = fallback

        if df.empty:
            return df, None, None

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        X = vectorizer.fit_transform(df["question"].tolist())
        logger.info(f"FloatChatbot: loaded KB with {len(df)} entries.")
        return df, vectorizer, X

    # ── KB lookup ────────────────────────────────────────────────────

    def _kb_lookup(self, query: str) -> tuple[str, float]:
        """Find the best KB answer via cosine similarity.

        Returns:
            (answer_text, confidence_score)
        """
        if self._vectorizer is None or self._kb_X is None or self._kb_df.empty:
            return "", 0.0

        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._kb_X).flatten()
        best_idx = int(np.argmax(sims))
        score = float(sims[best_idx])

        if score >= self.threshold:
            answer = self._kb_df.iloc[best_idx]["answer"]
            return answer, score

        return "", score   # empty string = no confident KB match

    # ── Public interface ─────────────────────────────────────────────

    def respond(self, query: str, df: pd.DataFrame | None = None) -> dict:
        """Process a user query and return a structured response.

        Routing order:
        1. Explanatory question patterns (what is / how does / who manages…)
           → try KB first; fall back to data engine or generic reply.
        2. Strong data-stat question (how many / total / distribution…)
           → go directly to data engine.
        3. Everything else
           → try KB; if no match, try data engine; then generic reply.

        Returns dict with keys: 'text', 'source', 'confidence'.
        """
        query = query.strip()
        if not query:
            return {"text": "Please type a question!", "source": "fallback", "confidence": 0.0}

        # ── Case 1: KB-priority patterns (explanatory questions) ──────
        if _has_kb_priority(query):
            answer, score = self._kb_lookup(query)
            if answer:
                return {"text": answer, "source": "knowledge_base", "confidence": score}
            # No good KB hit — try data engine as secondary
            if df is not None and not df.empty and _is_strong_data_query(query):
                return {
                    "text": self._data_engine.respond(query, df),
                    "source": "data", "confidence": 1.0,
                }
            # No data match either → polite fallback
            return {
                "text": (
                    "I don't have a specific answer for that yet. "
                    "Try asking about:\n"
                    "- 📊 ARGO profile counts, oceans, or institutions\n"
                    "- 🌊 What the ARGO project is\n"
                    "- 🛰️ How ARGO floats work or send data\n"
                    "- 🧪 What sensors ARGO floats carry"
                ),
                "source": "fallback", "confidence": 0.0,
            }

        # ── Case 2: Strong data questions → skip KB, go direct ────────
        if _is_strong_data_query(query) and df is not None and not df.empty:
            return {
                "text": self._data_engine.respond(query, df),
                "source": "data", "confidence": 1.0,
            }

        # ── Case 3: General query → try KB, then data, then fallback ──
        answer, score = self._kb_lookup(query)
        if answer:
            return {"text": answer, "source": "knowledge_base", "confidence": score}

        if df is not None and not df.empty:
            return {
                "text": self._data_engine.respond(query, df),
                "source": "data", "confidence": 0.7,
            }

        # Final fallback
        return {
            "text": (
                "I'm not sure about that. You can ask me about:\n"
                "- 🌊 The ARGO ocean observing programme\n"
                "- 📊 Profile counts, ocean distribution, institutions\n"
                "- 🛰️ How ARGO floats collect and transmit data\n"
                "- 🧪 Sensors, depth ranges, and data access"
            ),
            "source": "fallback", "confidence": 0.0,
        }

    @property
    def kb_size(self) -> int:
        """Number of entries in the knowledge base."""
        return len(self._kb_df)
