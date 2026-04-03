# modules/dashboard.py
# ─────────────────────────────────────────────────────────────────────────────
# Dashboard — All chart builders using Plotly with a consistent dark theme.
#
# Each function accepts a pd.DataFrame and returns a go.Figure object,
# ready to be passed into st.plotly_chart(fig, use_container_width=True).
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Shared dark-theme layout defaults
# ─────────────────────────────────────────────────────────────────────────────

_OCEAN_PALETTE = [
    "#00b4d8", "#0096c7", "#0077b6", "#023e8a",
    "#48cae4", "#90e0ef", "#ade8f4", "#caf0f8",
]

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,20,40,0.5)",
    font=dict(family="Inter, sans-serif", color="#c9d9e8"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        bgcolor="rgba(0,0,0,0.3)",
        bordercolor="rgba(0,180,216,0.2)",
        borderwidth=1,
    ),
    xaxis=dict(
        gridcolor="rgba(0,180,216,0.08)",
        zerolinecolor="rgba(0,180,216,0.15)",
    ),
    yaxis=dict(
        gridcolor="rgba(0,180,216,0.08)",
        zerolinecolor="rgba(0,180,216,0.15)",
    ),
)


def _apply_dark(fig: go.Figure) -> go.Figure:
    """Apply shared dark-theme layout to a figure."""
    fig.update_layout(**_DARK_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def plot_profiles_over_time(df: pd.DataFrame, freq: str = "ME") -> go.Figure:
    """Line chart with shaded fill showing profile count per month.

    Args:
        df:   ARGO profiles DataFrame with a 'timestamp' column.
        freq: pandas period alias ('ME' = month-end).

    Returns:
        Plotly Figure.
    """
    if "timestamp" not in df.columns or df.empty:
        return _empty_fig("No timestamp data available.")

    ts = df.groupby(df["timestamp"].dt.to_period(freq.rstrip("E"))).size()
    ts.index = ts.index.to_timestamp()
    ts_df = ts.reset_index()
    ts_df.columns = ["Date", "Profiles"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_df["Date"],
        y=ts_df["Profiles"],
        mode="lines",
        name="Profiles",
        line=dict(color="#00b4d8", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,180,216,0.12)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Profiles: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title="📈 Profiles Over Time",
        xaxis_title="Date",
        yaxis_title="Profile Count",
    )
    return _apply_dark(fig)


def plot_ocean_distribution(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of profiles per ocean.

    Args:
        df: ARGO profiles DataFrame with an 'ocean' column.

    Returns:
        Plotly Figure.
    """
    if "ocean" not in df.columns or df.empty:
        return _empty_fig("No ocean data available.")

    counts = df["ocean"].value_counts().head(8)
    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        marker=dict(
            color=counts.values,
            colorscale=[[0, "#023e8a"], [1, "#00b4d8"]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Profiles: %{x:,}<extra></extra>",
        text=[f"{v:,}" for v in counts.values],
        textposition="outside",
        textfont=dict(color="#90d5ec", size=11),
    ))
    fig.update_layout(
        title="🌊 Ocean Distribution",
        xaxis_title="Number of Profiles",
        yaxis=dict(autorange="reversed"),
    )
    return _apply_dark(fig)


def plot_profiler_types(df: pd.DataFrame) -> go.Figure:
    """Donut chart of profiler type distribution.

    Args:
        df: ARGO profiles DataFrame with a 'profiler_type' column.

    Returns:
        Plotly Figure.
    """
    if "profiler_type" not in df.columns or df.empty:
        return _empty_fig("No profiler_type data available.")

    counts = df["profiler_type"].value_counts().head(8)
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        marker=dict(colors=_OCEAN_PALETTE),
        hovertemplate="<b>%{label}</b><br>%{value:,} profiles<br>(%{percent})<extra></extra>",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title="⚙️ Profiler Type Distribution",
        showlegend=True,
    )
    return _apply_dark(fig)


def plot_top_institutions(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Vertical bar chart of the top contributing institutions.

    Args:
        df:    ARGO profiles DataFrame with an 'institution' column.
        top_n: how many institutions to show.

    Returns:
        Plotly Figure.
    """
    if "institution" not in df.columns or df.empty:
        return _empty_fig("No institution data available.")

    counts = df["institution"].value_counts().head(top_n)
    colors = px.colors.sample_colorscale(
        "Blues", [i / max(len(counts) - 1, 1) for i in range(len(counts))]
    )
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker=dict(color=list(reversed(colors))),
        hovertemplate="<b>%{x}</b><br>Profiles: %{y:,}<extra></extra>",
        text=[f"{v:,}" for v in counts.values],
        textposition="outside",
        textfont=dict(color="#90d5ec", size=10),
    ))
    fig.update_layout(
        title="🏢 Top Contributing Institutions",
        xaxis_title="Institution",
        yaxis_title="Profiles",
        xaxis=dict(tickangle=-35),
    )
    return _apply_dark(fig)


def plot_geo_scatter(df: pd.DataFrame, sample: int = 10_000) -> go.Figure:
    """World-map scatter plot of float locations.

    Randomly samples rows to keep the chart responsive.

    Args:
        df:     ARGO profiles DataFrame with 'latitude' and 'longitude'.
        sample: max rows to plot.

    Returns:
        Plotly Figure.
    """
    if not {"latitude", "longitude"}.issubset(df.columns) or df.empty:
        return _empty_fig("No lat/lon data available.")

    plot_df = df.dropna(subset=["latitude", "longitude"])
    if len(plot_df) > sample:
        plot_df = plot_df.sample(sample, random_state=42)

    hover_col = "ocean" if "ocean" in plot_df.columns else None
    fig = px.scatter_geo(
        plot_df,
        lat="latitude",
        lon="longitude",
        color=hover_col,
        color_discrete_sequence=_OCEAN_PALETTE,
        projection="natural earth",
        title="🗺️ ARGO Float Locations",
        opacity=0.5,
    )
    fig.update_geos(
        bgcolor="rgba(0,10,30,0.9)",
        landcolor="rgba(30,50,80,0.8)",
        oceancolor="rgba(0,30,60,0.7)",
        showocean=True,
        showland=True,
        showcoastlines=True,
        coastlinecolor="rgba(0,180,216,0.3)",
        showframe=False,
    )
    return _apply_dark(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _empty_fig(message: str) -> go.Figure:
    """Return a blank figure with a centred message — shown when data is absent."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#90d5ec"),
    )
    return _apply_dark(fig)
