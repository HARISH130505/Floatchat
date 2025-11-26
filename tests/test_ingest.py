# tests/test_ingest.py
import pandas as pd
from pathlib import Path

def test_csv_exists_and_columns():
    c = Path("data/profiles.csv")
    assert c.exists(), "CSV file missing: data/profiles.csv"
    df = pd.read_csv(c)

    # required columns
    for col in ['id', 'date', 'latitude', 'longitude', 'depth', 'temperature', 'salinity']:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 0

def test_ranges_and_types():
    df = pd.read_csv("data/profiles.csv")

    # date column as datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    assert pd.api.types.is_datetime64_any_dtype(df['date']), "date is not datetime64"

    # temperature & salinity ranges
    if "temperature" in df:
        assert df['temperature'].dropna().between(-5, 40).all(), "Temperature out of range"
    if "salinity" in df:
        assert df['salinity'].dropna().between(0, 50).all(), "Salinity out of range"

def test_no_duplicates():
    df = pd.read_csv("data/profiles.csv")
    if set(['id', 'date', 'depth']).issubset(df.columns):
        assert not df.duplicated(subset=['id', 'date', 'depth']).any(), "Duplicate rows found"
