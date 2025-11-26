# ingest/preview_profiles.py
import pandas as pd
import plotly.express as px
from pathlib import Path

# Point to CSV instead of parquet
C = Path("../data/profiles.csv")
if not C.exists():
    raise FileNotFoundError("Run the ingest pipeline first to create ../data/profiles.csv")

df = pd.read_csv(C)
Path("../data/plots").mkdir(parents=True, exist_ok=True)

# For now, visualize locations since profiles.csv does not contain per-depth temp
fig = px.scatter_geo(
    df,
    lat="latitude",
    lon="longitude",
    hover_name="file",
    title="Argo Profile Locations"
)
out = "../data/plots/profile_locations.html"
fig.write_html(out)
print("Wrote:", out)
