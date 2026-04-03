# modules/anomaly.py
# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection Module
#
# Supports two methods:
#   - Z-Score:         simple, fast, best for approximately normal distributions.
#   - Isolation Forest: ensemble tree method, works on multi-dimensional data.
#
# Since the ARGO index CSV only has metadata (no per-depth measurements),
# anomalies are detected on:
#   • Monthly profile *count* (temporal anomalies — e.g. a month with unusually
#     few/many reports)
#   • Spatial lat/lon distribution (geographic outliers)
#
# If a column named 'temperature' or 'salinity' exists, it will be used instead.
# ─────────────────────────────────────────────────────────────────────────────

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_flag(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask — True where |Z-score| > threshold."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    return ((series - mean).abs() / std) > threshold


def _isolation_forest_flag(df: pd.DataFrame, cols: list[str], contamination: float = 0.05) -> pd.Series:
    """Return boolean mask using Isolation Forest on the specified columns."""
    from sklearn.ensemble import IsolationForest

    sub = df[cols].dropna()
    if sub.empty or len(sub) < 20:
        return pd.Series(False, index=df.index)

    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = model.fit_predict(sub)                  # -1 = anomaly, 1 = normal
    mask = pd.Series(False, index=df.index)
    mask.loc[sub.index] = preds == -1
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Main detection function
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    method: str = "zscore",
    target: str = "count",
    zscore_threshold: float = 3.0,
    contamination: float = 0.05,
) -> dict:
    """Detect anomalies in ARGO data and return results for the UI.

    Args:
        df:               ARGO profiles DataFrame (QC-filtered).
        method:           'zscore' or 'isolation_forest'.
        target:           'count' (monthly counts) or 'spatial' (lat/lon).
        zscore_threshold: Z-score cutoff for the zscore method.
        contamination:    Fraction of expected anomalies for Isolation Forest.

    Returns:
        Dict with keys:
          - 'anomaly_count': int
          - 'total':         int
          - 'method':        str
          - 'target':        str
          - 'figure':        Plotly Figure
          - 'summary':       human-readable text explanation
          - 'flagged_df':    DataFrame of flagged rows / periods
    """
    if df is None or df.empty:
        return _empty_result("No data available for anomaly detection.")

    if target == "count":
        return _count_anomalies(df, method, zscore_threshold, contamination)
    elif target == "spatial":
        return _spatial_anomalies(df, method, zscore_threshold, contamination)
    else:
        return _empty_result(f"Unknown target: {target!r}. Choose 'count' or 'spatial'.")


# ─────────────────────────────────────────────────────────────────────────────
# Target: monthly profile count
# ─────────────────────────────────────────────────────────────────────────────

def _count_anomalies(df, method, threshold, contamination) -> dict:
    """Detect anomalous months in the profile-count time series."""
    if "timestamp" not in df.columns:
        return _empty_result("Need a 'timestamp' column for count-based anomaly detection.")

    ts = df.groupby(df["timestamp"].dt.to_period("M")).size()
    ts.index = ts.index.to_timestamp()
    ts_df = pd.DataFrame({"Date": ts.index, "Count": ts.values})

    if method == "zscore":
        ts_df["is_anomaly"] = _zscore_flag(ts_df["Count"], threshold)
    else:
        ts_df["is_anomaly"] = _isolation_forest_flag(ts_df, ["Count"], contamination)

    flagged = ts_df[ts_df["is_anomaly"]]
    n_anom = int(ts_df["is_anomaly"].sum())

    # ── Build figure ──────────────────────────────────────────────────
    fig = go.Figure()
    normal = ts_df[~ts_df["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=normal["Date"], y=normal["Count"],
        mode="lines+markers",
        name="Normal",
        line=dict(color="#00b4d8", width=1.5),
        marker=dict(size=4, color="#00b4d8"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Profiles: %{y:,}<extra></extra>",
    ))
    if not flagged.empty:
        fig.add_trace(go.Scatter(
            x=flagged["Date"], y=flagged["Count"],
            mode="markers",
            name="⚠️ Anomaly",
            marker=dict(size=10, color="#ff6b6b", symbol="x", line=dict(width=2, color="#ff0000")),
            hovertemplate="<b>%{x|%b %Y}</b><br>⚠️ Anomalous count: %{y:,}<extra></extra>",
        ))
    fig.update_layout(
        title="📊 Monthly Profile Count — Anomaly Detection",
        xaxis_title="Month", yaxis_title="Profile Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,20,40,0.5)",
        font=dict(family="Inter, sans-serif", color="#c9d9e8"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )

    method_label = "Z-Score (threshold=%.1f)" % threshold if method == "zscore" else "Isolation Forest"
    summary = (
        f"**Method**: {method_label}\n\n"
        f"**Target**: Monthly profile counts ({len(ts_df)} months analysed)\n\n"
        f"**Anomalies found**: **{n_anom}** month{'s' if n_anom != 1 else ''} "
        f"with unusual profile counts "
        f"({'none detected ✅' if n_anom == 0 else '⚠️ see highlighted points'})\n\n"
        + (
            f"**Flagged months**: {', '.join(flagged['Date'].dt.strftime('%b %Y').tolist())}"
            if n_anom > 0 else ""
        )
    )

    return {
        "anomaly_count": n_anom,
        "total": len(ts_df),
        "method": method_label,
        "target": "Monthly Count",
        "figure": fig,
        "summary": summary,
        "flagged_df": flagged[["Date", "Count"]].rename(columns={"Date": "Month", "Count": "Profile Count"}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Target: lat/lon spatial outliers
# ─────────────────────────────────────────────────────────────────────────────

def _spatial_anomalies(df, method, threshold, contamination) -> dict:
    """Detect geographically unusual float locations."""
    needed = {"latitude", "longitude"}
    if not needed.issubset(df.columns):
        return _empty_result("Need 'latitude' and 'longitude' columns for spatial anomaly detection.")

    sub = df[["latitude", "longitude"]].dropna().copy()

    if method == "zscore":
        lat_flag = _zscore_flag(sub["latitude"], threshold)
        lon_flag = _zscore_flag(sub["longitude"], threshold)
        sub["is_anomaly"] = lat_flag | lon_flag
    else:
        sub["is_anomaly"] = _isolation_forest_flag(sub, ["latitude", "longitude"], contamination)

    n_anom = int(sub["is_anomaly"].sum())
    normal = sub[~sub["is_anomaly"]]
    flagged = sub[sub["is_anomaly"]]

    # ── Build map figure ─────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=normal["latitude"], lon=normal["longitude"],
        mode="markers",
        name="Normal",
        marker=dict(size=3, color="#00b4d8", opacity=0.4),
        hovertemplate="Lat: %{lat:.2f}, Lon: %{lon:.2f}<extra>Normal</extra>",
    ))
    if not flagged.empty:
        fig.add_trace(go.Scattergeo(
            lat=flagged["latitude"], lon=flagged["longitude"],
            mode="markers",
            name="⚠️ Anomaly",
            marker=dict(size=8, color="#ff6b6b", symbol="x"),
            hovertemplate="Lat: %{lat:.2f}, Lon: %{lon:.2f}<extra>⚠️ Anomaly</extra>",
        ))
    fig.update_geos(
        bgcolor="rgba(0,10,30,0.9)",
        landcolor="rgba(30,50,80,0.8)",
        oceancolor="rgba(0,30,60,0.7)",
        showocean=True, showland=True,
        showcoastlines=True, coastlinecolor="rgba(0,180,216,0.3)",
        projection_type="natural earth",
    )
    fig.update_layout(
        title="🗺️ Spatial Anomaly Detection",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#c9d9e8"),
    )

    method_label = "Z-Score (threshold=%.1f)" % threshold if method == "zscore" else "Isolation Forest"
    summary = (
        f"**Method**: {method_label}\n\n"
        f"**Target**: Geographic lat/lon of {len(sub):,} floats\n\n"
        f"**Anomalies found**: **{n_anom:,}** spatially unusual position{'s' if n_anom != 1 else ''} "
        f"({'none detected ✅' if n_anom == 0 else '⚠️ see map'})"
    )

    return {
        "anomaly_count": n_anom,
        "total": len(sub),
        "method": method_label,
        "target": "Spatial (lat/lon)",
        "figure": fig,
        "summary": summary,
        "flagged_df": flagged[["latitude", "longitude"]].reset_index(drop=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _empty_result(msg: str) -> dict:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14, color="#90d5ec"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d9e8"))
    return {
        "anomaly_count": 0, "total": 0, "method": "N/A", "target": "N/A",
        "figure": fig, "summary": msg, "flagged_df": pd.DataFrame(),
    }
