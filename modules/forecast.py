# modules/forecast.py
# ─────────────────────────────────────────────────────────────────────────────
# Forecasting Module
#
# Predicts future ARGO profile activity using:
#   1. Linear Regression (sklearn) — always available, fast, interpretable.
#   2. ARIMA (statsmodels)         — better for seasonal/autocorrelated data;
#                                    gracefully skipped if statsmodels missing.
#
# Input:  ARGO profiles DataFrame with a 'timestamp' column.
# Output: Dict with Plotly Figure + textual summary.
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared dark layout
# ─────────────────────────────────────────────────────────────────────────────

_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,20,40,0.5)",
    font=dict(family="Inter, sans-serif", color="#c9d9e8"),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(0,180,216,0.2)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(0,180,216,0.08)", zerolinecolor="rgba(0,180,216,0.15)"),
    yaxis=dict(gridcolor="rgba(0,180,216,0.08)", zerolinecolor="rgba(0,180,216,0.15)"),
    margin=dict(l=10, r=10, t=50, b=20),
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ARGO profiles into a monthly count series.

    Returns:
        DataFrame with columns ['ds', 'y'] where:
          ds = datetime (month start)
          y  = profile count.
    """
    ts = df.groupby(df["timestamp"].dt.to_period("M")).size()
    ts.index = ts.index.to_timestamp()
    return pd.DataFrame({"ds": ts.index, "y": ts.values}).reset_index(drop=True)


def _future_dates(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    """Generate `periods` monthly timestamps after last_date."""
    return pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq="MS")


# ─────────────────────────────────────────────────────────────────────────────
# Linear Regression Forecast
# ─────────────────────────────────────────────────────────────────────────────

def _linear_forecast(hist_df: pd.DataFrame, periods: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit LinearRegression on ordinal timestamps and extrapolate.

    Returns:
        (in_sample_preds, out_of_sample_preds) as numpy arrays.
    """
    X = hist_df["ds"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = hist_df["y"].values

    model = LinearRegression()
    model.fit(X, y)

    future = _future_dates(hist_df["ds"].iloc[-1], periods)
    X_fut = pd.DatetimeIndex(future).map(pd.Timestamp.toordinal).values.reshape(-1, 1)

    in_sample  = np.clip(model.predict(X), 0, None)         # can't have negative counts
    out_sample = np.clip(model.predict(X_fut), 0, None)
    return in_sample, out_sample, future, model.coef_[0]


# ─────────────────────────────────────────────────────────────────────────────
# ARIMA Forecast (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _arima_forecast(hist_df: pd.DataFrame, periods: int, order=(1, 1, 1)):
    """Fit ARIMA and return out-of-sample forecast.

    Returns None if statsmodels is unavailable or fitting fails.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(hist_df["y"].values, order=order)
        result = model.fit()
        forecast_obj = result.get_forecast(steps=periods)
        mean = forecast_obj.predicted_mean
        ci   = forecast_obj.conf_int(alpha=0.2)          # 80 % confidence interval
        future_dates = _future_dates(hist_df["ds"].iloc[-1], periods)
        return mean, ci, future_dates
    except ImportError:
        logger.info("statsmodels not installed — ARIMA skipped.")
        return None
    except Exception as exc:
        logger.warning(f"ARIMA fitting failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def forecast_trend(
    df: pd.DataFrame,
    periods: int = 12,
    method: str = "linear",
    arima_order: tuple = (1, 1, 1),
) -> dict:
    """Forecast future ARGO profile counts.

    Args:
        df:          ARGO profiles DataFrame with a 'timestamp' column.
        periods:     number of future months to forecast.
        method:      'linear' or 'arima'.
        arima_order: (p, d, q) order for ARIMA (ignored for linear).

    Returns:
        Dict with keys:
          - 'figure':    Plotly Figure
          - 'summary':   text description
          - 'forecast_df': DataFrame of future predicted counts
          - 'method':    method name string
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return _empty_result("No timestamp data available for forecasting.")

    hist_df = _aggregate_monthly(df)

    if len(hist_df) < 6:
        return _empty_result("Not enough historical data (need ≥ 6 months) to forecast.")

    fig = go.Figure()

    # ── Historical trace (always shown) ─────────────────────────────
    fig.add_trace(go.Scatter(
        x=hist_df["ds"], y=hist_df["y"],
        mode="lines",
        name="Historical",
        line=dict(color="#00b4d8", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,180,216,0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Profiles: %{y:,.0f}<extra>Historical</extra>",
    ))

    forecast_values = None
    future_dates = None
    method_label = method.title()
    trend_note = ""

    # ── Linear regression ────────────────────────────────────────────
    if method == "linear":
        in_sample, out_sample, future_dates, coef = _linear_forecast(hist_df, periods)
        forecast_values = out_sample
        direction = "upward 📈" if coef > 0 else "downward 📉"
        trend_note = f"Linear trend is **{direction}** ({coef:+.1f} profiles/month on average)."
        method_label = "Linear Regression"

        # In-sample fit line (dashed)
        fig.add_trace(go.Scatter(
            x=hist_df["ds"], y=in_sample,
            mode="lines",
            name="Trend (fit)",
            line=dict(color="#48cae4", width=1.5, dash="dot"),
            hoverinfo="skip",
        ))

    # ── ARIMA ────────────────────────────────────────────────────────
    elif method == "arima":
        result = _arima_forecast(hist_df, periods, arima_order)
        if result is None:
            # Fall back to linear silently
            in_sample, out_sample, future_dates, coef = _linear_forecast(hist_df, periods)
            forecast_values = out_sample
            method_label = "Linear Regression (ARIMA fallback)"
            trend_note = "ARIMA was unavailable or failed — switched to linear regression."
        else:
            mean, ci, future_dates = result
            forecast_values = mean
            method_label = f"ARIMA{arima_order}"
            trend_note = "ARIMA confidence interval (80%) shown as shaded band."

            # Confidence band
            ci_lower = np.clip(ci[:, 0], 0, None)
            ci_upper = np.clip(ci[:, 1], 0, None)
            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(ci_upper) + list(ci_lower[::-1]),
                fill="toself",
                fillcolor="rgba(0,150,180,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="80% CI",
                hoverinfo="skip",
            ))

    # ── Forecast trace ───────────────────────────────────────────────
    if forecast_values is not None and future_dates is not None:
        # Connector point (last historical → first forecast)
        connector_x = [hist_df["ds"].iloc[-1], future_dates[0]]
        connector_y = [hist_df["y"].iloc[-1], float(forecast_values[0])]
        fig.add_trace(go.Scatter(
            x=connector_x, y=connector_y,
            mode="lines",
            line=dict(color="#ff9f1c", width=1.5, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=forecast_values,
            mode="lines+markers",
            name=f"Forecast ({periods} mo.)",
            line=dict(color="#ff9f1c", width=2, dash="dash"),
            marker=dict(size=6, color="#ff9f1c"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra>Forecast</extra>",
        ))

    fig.update_layout(
        title=f"📉 Profile Count Forecast — {method_label}",
        xaxis_title="Date",
        yaxis_title="Profiles per Month",
        **_DARK,
    )

    # ── Build forecast DataFrame ─────────────────────────────────────
    if forecast_values is not None and future_dates is not None:
        fcast_df = pd.DataFrame({
            "Month": pd.DatetimeIndex(future_dates).strftime("%b %Y"),
            "Predicted Profiles": np.round(forecast_values).astype(int),
        })
    else:
        fcast_df = pd.DataFrame()

    summary = (
        f"**Method**: {method_label}\n\n"
        f"**Historical data**: {len(hist_df)} months "
        f"({hist_df['ds'].iloc[0].strftime('%b %Y')} → {hist_df['ds'].iloc[-1].strftime('%b %Y')})\n\n"
        f"**Forecast horizon**: {periods} months\n\n"
        + (f"{trend_note}" if trend_note else "")
    )

    return {
        "figure": fig,
        "summary": summary,
        "forecast_df": fcast_df,
        "method": method_label,
    }


def _empty_result(msg: str) -> dict:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14, color="#90d5ec"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d9e8"))
    return {"figure": fig, "summary": msg, "forecast_df": pd.DataFrame(), "method": "N/A"}
