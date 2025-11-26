# ingest/load_to_postgres.py
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

C = Path("../data/profiles.csv")   # <-- changed to CSV
if not C.exists():
    raise FileNotFoundError(f"CSV not found: {C}")

PG_CONN = os.getenv("PG_CONN")
if not PG_CONN:
    raise EnvironmentError("Set PG_CONN environment variable e.g. postgresql://user:pass@host:5432/db")

engine = create_engine(PG_CONN)

# Load CSV
df = pd.read_csv(C)

# Load into Postgres
df.to_sql("floats", engine, if_exists="replace", index=False, method='multi', chunksize=50000)

# Create indexes
with engine.begin() as conn:
    if "date" in df.columns:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_date ON floats(date);"))
    if {"latitude", "longitude"}.issubset(df.columns):
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_lat_lon ON floats(latitude, longitude);"))

print("✅ Loaded CSV data into Postgres")
