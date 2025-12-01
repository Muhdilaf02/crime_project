# ================================================================
# utils/data_pipeline.py — load & normalize dataset
# ================================================================
from __future__ import annotations
import pandas as pd
import numpy as np
from .common import get_paths, log

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip().str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return df

def _ensure_total_crimes(df: pd.DataFrame) -> pd.DataFrame:
    """Jika tiada total_crimes_reported atau semuanya NaN, kira dari sub-kategori jika ada.
       Jika dataset adalah per-laporan (setiap baris satu kes), kita boleh gunakan kiraan groupby di halaman.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    # cuba guna kolum standard
    if "total_crimes_reported" in df.columns:
        df["total_crimes_reported"] = pd.to_numeric(df["total_crimes_reported"], errors="coerce")
    else:
        df["total_crimes_reported"] = np.nan

    if df["total_crimes_reported"].fillna(0).sum() == 0:
        # fallback: jumlahkan sub-kategori jika wujud
        parts = [c for c in ["violent_crimes","property_crimes","cyber_crimes"] if c in df.columns]
        if parts:
            df["total_crimes_reported"] = df[parts].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        else:
            # biarkan NaN; halaman akan guna count per group (baris = 1 jenayah)
            pass
    return df

def load_cleaned_data() -> pd.DataFrame:
    """Baca dataset projek sebenar dari ../output/final_cleaned_crime_socioeconomic_data.csv"""
    p = get_paths()["OUTPUT"] / "final_cleaned_crime_socioeconomic_data.csv"
    if not p.exists():
        raise FileNotFoundError(f"Dataset tidak dijumpai di: {p}")
    df = pd.read_csv(p)
    df = _normalize_columns(df)

    # parse tarikh jika wujud
    for c in ["date_reported","date_of_occurrence","date_case_closed"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # pastikan numeric untuk beberapa kolum penting
    for c in ["victim_age","violent_crimes","property_crimes","cyber_crimes","population"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # pastikan year wujud
    if "year" not in df.columns:
        if "date_reported" in df.columns:
            df["year"] = df["date_reported"].dt.year
        else:
            df["year"] = np.nan

    # pastikan 'state' unified field
    if "state_name" in df.columns:
        df["state"] = df["state_name"].astype(str)
    elif "state_x" in df.columns:
        df["state"] = df["state_x"].astype(str)

    # create/repair total_crimes_reported if applicable
    df = _ensure_total_crimes(df)

    log().info(f"Loaded final dataset: {len(df):,} rows")
    return df

def numeric_defaults(df: pd.DataFrame) -> dict[str, float]:
    """Median untuk semua numeric (diguna untuk isi feature hilang ketika prediction)."""
    s = df.select_dtypes(include=["number"]).median(numeric_only=True)
    return {k: float(v) for k, v in s.fillna(0).to_dict().items()}
