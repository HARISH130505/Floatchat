# modules/qc_guard.py
# ─────────────────────────────────────────────────────────────────────────────
# QC-Guard Layer
#
# ARGO floats assign quality-control flags to each measurement:
#   1 = Good data
#   2 = Probably good data
#   3 = Probably bad data
#   4 = Bad data
#   9 = Missing value
#
# This module wraps every data query result and keeps only flag 1 & 2 rows.
# If the 'value_qc' column is absent (common in index-only CSVs), the filter
# is skipped gracefully and a note is displayed.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

# QC flags considered "good quality"
GOOD_QC_FLAGS = {1, 2}


def apply_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Filter a DataFrame to retain only ARGO quality-controlled good data.

    Args:
        df: Raw profiles DataFrame, possibly containing a 'value_qc' column.

    Returns:
        Tuple of:
          - filtered DataFrame (or original if 'value_qc' not present)
          - status string describing what was done (for UI display)
    """
    if df.empty:
        return df, "⚠️ No data to apply QC filter."

    if "value_qc" not in df.columns:
        # Index-only CSVs don't carry per-measurement QC flags — skip gracefully.
        return df, (
            "ℹ️ **QC Guard**: `value_qc` column not found in this dataset — "
            "QC filtering skipped. All rows retained."
        )

    before = len(df)
    # Coerce to numeric; non-numeric flags → NaN → treated as unknown, dropped.
    df = df.copy()
    df["value_qc"] = pd.to_numeric(df["value_qc"], errors="coerce")
    filtered = df[df["value_qc"].isin(GOOD_QC_FLAGS)]
    after = len(filtered)
    pct = 100 * after / before if before > 0 else 0

    status = (
        f"✅ **QC Guard Active** — Retained **{after:,}** of **{before:,}** profiles "
        f"({pct:.1f}%) with quality flags 1 (Good) & 2 (Probably Good)."
    )
    return filtered.reset_index(drop=True), status


def qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a breakdown of QC flag distribution in the dataset.

    Useful for displaying a small info table in the UI.
    """
    if "value_qc" not in df.columns:
        return pd.DataFrame({"Note": ["value_qc column not present"]})

    flag_map = {
        1: "Good data",
        2: "Probably good data",
        3: "Probably bad data",
        4: "Bad data",
        9: "Missing value",
    }
    counts = df["value_qc"].value_counts().reset_index()
    counts.columns = ["Flag", "Count"]
    counts["Meaning"] = counts["Flag"].map(flag_map).fillna("Unknown")
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(2).astype(str) + "%"
    return counts[["Flag", "Meaning", "Count", "Percentage"]].sort_values("Flag")
