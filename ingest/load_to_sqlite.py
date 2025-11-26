# ingest/load_to_sqlite.py
import sqlite3
from pathlib import Path
import pandas as pd

C = Path("../data/profiles.csv")
if not C.exists():
    raise FileNotFoundError(f"CSV not found: {C}")

df = pd.read_csv(C)
conn = sqlite3.connect("../data/argo_poc.db")
df.to_sql("floats", conn, if_exists="replace", index=False)

# Fix column names to match CSV (latitude/longitude)
conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON floats(date);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_lat ON floats(latitude);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_lat_lon ON floats(latitude, longitude);")

conn.commit()
conn.close()
print("✅ Loaded to sqlite: data/argo_poc.db")