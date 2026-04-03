# modules/__init__.py
# FloatChat — Module Package
# Exposes all sub-modules for clean top-level imports.

from modules.database  import DataManager
from modules.qc_guard  import apply_qc
from modules.chatbot   import FloatChatbot
from modules.dashboard import (
    plot_profiles_over_time,
    plot_ocean_distribution,
    plot_profiler_types,
    plot_top_institutions,
)
from modules.anomaly   import detect_anomalies
from modules.forecast  import forecast_trend

__all__ = [
    "DataManager",
    "apply_qc",
    "FloatChatbot",
    "plot_profiles_over_time",
    "plot_ocean_distribution",
    "plot_profiler_types",
    "plot_top_institutions",
    "detect_anomalies",
    "forecast_trend",
]
