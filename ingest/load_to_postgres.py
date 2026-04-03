# ingest/load_to_postgres.py
import os
import time
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

# Map of CSV filename -> Target PostgreSQL table name
DATA_FILES = {
    "argo_all.csv": "argo_all",
    "profiles.csv": "profiles",
    "metadata.csv": "metadata",
    "knowledge_base.csv": "knowledge_base",
}

def load_data():
    pg_conn = os.getenv("PG_CONN")
    if not pg_conn:
        raise EnvironmentError(
            "Set PG_CONN environment variable e.g. postgresql://user:pass@host:5432/db"
        )
    
    engine = create_engine(pg_conn)
    base_dir = Path(__file__).parent.parent / "data"

    print(f"Connecting to PostgreSQL...")
    
    for filename, table_name in DATA_FILES.items():
        csv_path = base_dir / filename
        if not csv_path.exists():
            print(f"⚠️  Skipping {filename} — file not found in data/.")
            continue
            
        print(f"📦 Loading {filename} into table '{table_name}'...")
        start_t = time.time()
        
        # argo_all is huge, read in chunks if necessary, but pandas to_sql chunks it anyway
        df = pd.read_csv(csv_path)
        
        # Basic datetime parsing for indexing
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            
        # For metadata.csv/profiles.csv, ensure float_id exists since we join on it
        if "file" in df.columns and table_name in ["profiles", "metadata"]:
            df["float_id"] = df["file"].astype(str).str.split("/").str[1]
            
        # Upload
        df.to_sql(table_name, engine, if_exists="replace", index=False, chunksize=50000, method="multi")
        
        # Create indexes
        with engine.begin() as conn:
            if "date" in df.columns:
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date);"))
            if {"latitude", "longitude"}.issubset(df.columns):
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_lat_lon ON {table_name}(latitude, longitude);"))
            if "float_id" in df.columns:
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_float_id ON {table_name}(float_id);"))
                
        print(f"✅ Finished {table_name} in {time.time() - start_t:.1f}s ({len(df):,} rows)")
        
    print("🚀 All datasets loaded successfully!")

if __name__ == "__main__":
    load_data()
