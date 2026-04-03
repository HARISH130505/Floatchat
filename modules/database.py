# modules/database.py
# ─────────────────────────────────────────────────────────────────────────────
# DataManager: Provides a unified data-access layer.
#
# Strategy:
#   1. Try PostgreSQL (via PG_CONN env var) first — production mode.
#   2. Fall back to loading data/argo_all.csv in memory — development mode.
#
# All public methods return pd.DataFrame so callers are DB-agnostic.
# ─────────────────────────────────────────────────────────────────────────────

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ocean code → full name
OCEAN_NAMES = {"P": "Pacific Ocean", "A": "Atlantic Ocean", "I": "Indian Ocean"}

# Path to the CSV fallback
_CSV_PATH      = Path(__file__).parent.parent / "data" / "argo_all.csv"
_PROFILES_PATH = Path(__file__).parent.parent / "data" / "profiles.csv"
_METADATA_PATH = Path(__file__).parent.parent / "data" / "metadata.csv"


class DataManager:
    """Unified data access layer for ARGO float data.

    Usage:
        dm = DataManager()
        df = dm.get_profiles()          # full dataset (QC not applied here)
        summary = dm.get_summary_stats(df)
    """

    def __init__(self):
        self._engine = None
        self._csv_cache: pd.DataFrame | None = None
        self._meta_cache: pd.DataFrame | None = None   # merged profiles + metadata
        self._mode = "csv"          # "postgres" | "csv"
        self._connect()

    # ──────────────────────────────────────────────────────────────────
    # Internal connection logic
    # ──────────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Try PostgreSQL from st.secrets or env var; fall back to CSV silently."""
        import streamlit as st
        
        pg_conn = ""
        # 1. Try Streamlit Secrets
        try:
            pg_conn = st.secrets.get("PG_CONN", "")
        except FileNotFoundError:
            pass
            
        # 2. Try OS Environment
        if not pg_conn:
            pg_conn = os.getenv("PG_CONN", "").strip()
            
        if pg_conn:
            try:
                from sqlalchemy import create_engine, text

                engine = create_engine(pg_conn, pool_pre_ping=True)
                # Quick connectivity test
                with engine.connect() as c:
                    c.execute(text("SELECT 1"))
                self._engine = engine
                self._mode = "postgres"
                logger.info("DataManager: connected to PostgreSQL ✅")
            except Exception as exc:
                logger.warning(f"DataManager: PostgreSQL unavailable ({exc}), falling back to CSV.")
        else:
            logger.info("DataManager: PG_CONN not set — using CSV fallback.")

    # ──────────────────────────────────────────────────────────────────
    # CSV fallback helpers
    # ──────────────────────────────────────────────────────────────────

    def _load_csv(self) -> pd.DataFrame:
        """Load and cache the CSV once per session."""
        if self._csv_cache is None:
            if not _CSV_PATH.exists():
                logger.error(f"CSV not found at {_CSV_PATH}")
                return pd.DataFrame()
            df = pd.read_csv(_CSV_PATH)
            df = self._normalise(df)
            self._csv_cache = df
        return self._csv_cache.copy()

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column types after loading."""
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
        # Map 1-char ocean codes → full names
        if "ocean" in df.columns:
            df["ocean"] = df["ocean"].astype(str).map(OCEAN_NAMES).fillna(df["ocean"])
        for col in ("profiler_type", "institution"):
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df

    # ──────────────────────────────────────────────────────────────────
    # PostgreSQL helpers
    # ──────────────────────────────────────────────────────────────────

    def _pg_query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        """Run a raw SQL string and return a DataFrame."""
        from sqlalchemy import text

        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Return 'postgres' or 'csv'."""
        return self._mode

    def get_profiles(self, limit: int | None = None) -> pd.DataFrame:
        """Return all (or limited) profile rows from argo_all."""
        if self._mode == "postgres":
            cap = f"LIMIT {int(limit)}" if limit else ""
            sql = f"""
                SELECT date AS timestamp,
                       latitude, longitude,
                       ocean, profiler_type, institution,
                       value_qc
                  FROM argo_all
                 {cap}
            """
            return self._normalise(self._pg_query(sql))
        return self._load_csv()

    def get_knowledge_base(self) -> pd.DataFrame:
        """Fetch Knowledge Base Q&A from Postgres or CSV fallback."""
        if self._mode == "postgres":
            try:
                return self._pg_query("SELECT question, answer FROM knowledge_base")
            except Exception as e:
                logger.warning(f"Failed to query knowledge_base table: {e}. Falling back to CSV.")

        kb_path = Path(__file__).parent.parent / "data" / "knowledge_base.csv"
        if kb_path.exists():
            return pd.read_csv(kb_path)
        return pd.DataFrame()

    def get_summary_stats(self, df: pd.DataFrame) -> dict:
        """Compute high-level summary statistics from a DataFrame.

        Args:
            df: ARGO profiles DataFrame (already QC-filtered if desired).

        Returns:
            Dict with keys: total_profiles, unique_oceans, unique_institutions,
            date_min, date_max.
        """
        stats: dict = {}
        stats["total_profiles"] = len(df)
        stats["unique_oceans"] = df["ocean"].nunique() if "ocean" in df.columns else 0
        stats["unique_institutions"] = (
            df["institution"].nunique() if "institution" in df.columns else 0
        )
        if "timestamp" in df.columns and not df["timestamp"].isna().all():
            stats["date_min"] = df["timestamp"].min().strftime("%Y-%m-%d")
            stats["date_max"] = df["timestamp"].max().strftime("%Y-%m-%d")
        else:
            stats["date_min"] = stats["date_max"] = "N/A"
        return stats

    def get_time_series(self, df: pd.DataFrame, freq: str = "ME") -> pd.Series:
        """Aggregate profiles by time period.

        Args:
            df: profiles DataFrame.
            freq: pandas offset alias, e.g. 'ME' (month-end), 'QE' (quarter).

        Returns:
            Series indexed by period timestamp, values = profile count.
        """
        if "timestamp" not in df.columns or df.empty:
            return pd.Series(dtype=int)
        ts = df.groupby(df["timestamp"].dt.to_period(freq.rstrip("E"))).size()
        ts.index = ts.index.to_timestamp()
        return ts

    def get_by_ocean(self, df: pd.DataFrame) -> pd.Series:
        """Count profiles per ocean."""
        if "ocean" not in df.columns:
            return pd.Series(dtype=int)
        return df["ocean"].value_counts()

    def get_by_institution(self, df: pd.DataFrame, top_n: int = 10) -> pd.Series:
        """Count profiles per institution (top N)."""
        if "institution" not in df.columns:
            return pd.Series(dtype=int)
        return df["institution"].value_counts().head(top_n)

    def get_by_profiler(self, df: pd.DataFrame) -> pd.Series:
        """Count profiles per profiler type."""
        if "profiler_type" not in df.columns:
            return pd.Series(dtype=int)
        return df["profiler_type"].value_counts()

    @property
    def has_enriched_data(self) -> bool:
        """True if both profiles.csv and metadata.csv are available on disk."""
        return _PROFILES_PATH.exists() and _METADATA_PATH.exists()

    def get_merged_profiles(self) -> pd.DataFrame:
        """Load and merge profiles + metadata into an enriched DataFrame."""
        if self._meta_cache is not None:
            return self._meta_cache.copy()

        if self._mode == "postgres":
            try:
                # Perform the JOIN entirely in PostgreSQL for massive speedup
                sql = """
                    SELECT p.*, m.date_update AS meta_date_update
                      FROM profiles p
                      LEFT JOIN metadata m ON p.float_id = m.float_id
                """
                merged = self._pg_query(sql)
                
                # Standardise
                if "date" in merged.columns:
                    merged = merged.rename(columns={"date": "timestamp"})
                if "timestamp" in merged.columns:
                    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
                
                if "date_update" in merged.columns and "meta_date_update" in merged.columns:
                    # Prefer metadata date_update, fallback to profile date_update
                    merged["date_update"] = pd.to_datetime(merged["meta_date_update"], errors="coerce").combine_first(
                        pd.to_datetime(merged["date_update"], errors="coerce")
                    )
                    merged = merged.drop(columns=["meta_date_update"])
                
                self._meta_cache = merged
                return merged.copy()
            except Exception as e:
                logger.warning(f"Failed to join profiles/metadata via SQL: {e}. Falling back to CSV.")

        # --- CSV FALLBACK ---
        if not _PROFILES_PATH.exists():
            logger.warning("profiles.csv not found — cannot build enriched dataset.")
            return pd.DataFrame()

        # ── Load profiles ─────────────────────────────────────────────
        prof = pd.read_csv(_PROFILES_PATH)
        prof = self._normalise(prof)          # standardises date → timestamp, ocean codes

        if _METADATA_PATH.exists():
            # ── Load metadata ─────────────────────────────────────────
            meta = pd.read_csv(_METADATA_PATH)

            # Extract the actual float ID from the file path ('kma/5900142/profiles/R5...nc' -> '5900142')
            prof["float_id"] = prof["file"].str.split("/").str[1]
            meta["float_id"] = meta["file"].str.split("/").str[1]

            # Parse date_update as datetime
            if "date_update" in meta.columns:
                meta["date_update"] = pd.to_datetime(
                    meta["date_update"], errors="coerce", format="%Y%m%d%H%M%S"
                )

            # Keep only the columns from metadata that aren't already in profiles, plus our join key
            meta_cols = ["float_id", "date_update"] + [
                c for c in meta.columns
                if c not in prof.columns and c not in ["file", "float_id"]
            ]
            meta = meta[[c for c in meta_cols if c in meta.columns]]
            meta = meta.drop_duplicates(subset=["float_id"])

            # Left-join so we keep all profiles even if no metadata match
            merged = prof.merge(meta, on="float_id", how="left")
            merged = merged.drop(columns=["float_id"])
            
            # If both had date_update, pandas creates date_update_x and date_update_y
            if "date_update_y" in merged.columns and "date_update_x" in merged.columns:
                merged["date_update"] = merged["date_update_y"].combine_first(merged["date_update_x"])
                merged = merged.drop(columns=["date_update_x", "date_update_y"])

        else:
            logger.info("metadata.csv not found — returning profiles only.")
            merged = prof

        logger.info(
            f"DataManager: merged enriched dataset — {len(merged):,} rows, "
            f"columns: {list(merged.columns)}"
        )
        self._meta_cache = merged
        return merged.copy()

    def get_float_activity(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Summarise per-float activity from the merged dataset.

        Args:
            merged_df: result of get_merged_profiles().

        Returns:
            DataFrame with one row per unique float file, showing:
            first_seen, last_seen, last_update, profile_count, ocean, institution.
        """
        if merged_df.empty or "file" not in merged_df.columns:
            return pd.DataFrame()

        agg: dict = {"profile_count": ("file", "count")}
        if "timestamp"    in merged_df.columns: agg["first_seen"]  = ("timestamp",    "min")
        if "timestamp"    in merged_df.columns: agg["last_seen"]   = ("timestamp",    "max")
        if "date_update"  in merged_df.columns: agg["last_update"] = ("date_update",  "max")
        if "ocean"        in merged_df.columns: agg["ocean"]       = ("ocean",        "first")
        if "institution"  in merged_df.columns: agg["institution"] = ("institution",  "first")

        summary = (
            merged_df.groupby("file")
            .agg(**agg)
            .reset_index()
            .sort_values("profile_count", ascending=False)
        )
        return summary

