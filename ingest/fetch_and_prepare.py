import argparse
import logging
from pathlib import Path
import pandas as pd
import requests

# ----------------------------
# Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Robust path resolving to floatchat-poc/data regardless of where script is run from
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ARGO_FILES = {
    "prof": "https://data-argo.ifremer.fr/ar_index_global_prof.txt",
    "meta": "https://data-argo.ifremer.fr/ar_index_global_meta.txt"
}


# ----------------------------
# Helpers
# ----------------------------
def download_file(url: str, dest: Path) -> Path:
    """Download file if not already cached locally."""
    if dest.exists():
        logging.info(f"✅ Using local {dest}")
        return dest

    logging.info(f"⬇️  Downloading {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    logging.info(f"✅ Downloaded to {dest}")
    return dest


def load_prof_index(index_file: Path) -> pd.DataFrame:
    """Load profile index into DataFrame."""
    logging.info(f"📂 Loading profile index from {index_file}")
    df = pd.read_csv(index_file, comment="#")
    df.columns = [
        "file", "date", "latitude", "longitude",
        "ocean", "profiler_type", "institution", "date_update"
    ]
    return df


def load_meta_index(index_file: Path) -> pd.DataFrame:
    """Load the metadata index file into a DataFrame."""
    logging.info(f"📂 Loading metadata index from {index_file}")
    df = pd.read_csv(index_file, comment="#", header=0)  # header row exists
    expected = ["file", "profiler_type", "institution", "date_update"]
    if len(df.columns) != len(expected):
        raise ValueError(f"Metadata index: expected {len(expected)} cols but found {len(df.columns)}")
    df.columns = expected
    return df


# ----------------------------
# Cleaning
# ----------------------------
def clean_profiles(df: pd.DataFrame, bbox=None) -> pd.DataFrame:
    """Clean profile dataframe before filtering."""
    n0 = len(df)
    df = df.drop_duplicates()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["file", "latitude", "longitude", "date"])
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    if bbox:
        lon_min, lon_max, lat_min, lat_max = bbox
        margin = 2.0  # degrees margin beyond bbox
        df = df[
            (df["longitude"].between(lon_min - margin, lon_max + margin)) &
            (df["latitude"].between(lat_min - margin, lat_max + margin))
        ]

    logging.info(f"🧹 Cleaned profiles: {n0} → {len(df)} rows")
    return df


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Clean metadata dataframe before filtering."""
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=["file", "date_update"])
    df["date_update"] = pd.to_datetime(df["date_update"], errors="coerce", format="%Y%m%d%H%M%S")
    df = df.dropna(subset=["date_update"])
    df = df.sort_values("date_update")

    logging.info(f"🧹 Cleaned metadata: {n0} → {len(df)} rows")
    return df


# ----------------------------
# Filters
# ----------------------------
def filter_profiles(df: pd.DataFrame, bbox=None, time_range=None) -> pd.DataFrame:
    """Filter profile dataframe by bounding box and/or time range."""
    if bbox:
        lon_min, lon_max, lat_min, lat_max = bbox
        df = df[
            (df["longitude"].between(lon_min, lon_max)) &
            (df["latitude"].between(lat_min, lat_max))
        ]
        logging.info(f"📍 Filtered profiles to {len(df)} rows in bbox")

    if time_range:
        start, end = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        logging.info(f"🗓️ Filtered profiles to {len(df)} rows in time range")

    return df


def filter_metadata(df: pd.DataFrame, time_range=None) -> pd.DataFrame:
    """Filter metadata dataframe by time range."""
    if time_range:
        start, end = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
        df = df[(df["date_update"] >= start) & (df["date_update"] <= end)]
        logging.info(f"🗓️ Filtered metadata to {len(df)} rows in time range")

    return df


# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(bbox=None, time_range=None):
    # Download/cached files
    prof_index = download_file(ARGO_FILES["prof"], DATA_DIR / "ar_index_global_prof.txt")
    meta_index = download_file(ARGO_FILES["meta"], DATA_DIR / "ar_index_global_meta.txt")

    # Load
    prof_df = load_prof_index(prof_index)
    meta_df = load_meta_index(meta_index)

    # Clean
    prof_df = clean_profiles(prof_df, bbox)
    meta_df = clean_metadata(meta_df)

    # Filter
    prof_df = filter_profiles(prof_df, bbox, time_range)
    meta_df = filter_metadata(meta_df, time_range)

    # Save
    prof_out = DATA_DIR / "profiles.csv"
    meta_out = DATA_DIR / "metadata.csv"
    prof_df.to_csv(prof_out, index=False)
    meta_df.to_csv(meta_out, index=False)

    logging.info(f"✅ Saved profiles to {prof_out}")
    logging.info(f"✅ Saved metadata to {meta_out}")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, help="Bounding box: lon_min lon_max lat_min lat_max")
    parser.add_argument("--time", nargs=2, help="Start and end dates (YYYY-MM-DD)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bbox = args.bbox if args.bbox else None
    time_range = tuple(args.time) if args.time else None
    run_pipeline(bbox=bbox, time_range=time_range)