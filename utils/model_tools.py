# ================================================================
# utils/model_tools.py — real-time models + prediction
# ================================================================
from __future__ import annotations
import json, os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from .common import get_paths, log
from .data_pipeline import load_cleaned_data, numeric_defaults

# -------------------------------------------
# Lokasi fail model yang dihasilkan oleh notebook
# (Notebook perlu save ke ../output/)
# -------------------------------------------
def model_paths() -> dict[str, str]:
    out = get_paths()["OUTPUT"]
    return {
        "reg": str(out / "regressor.pkl"),
        "cls": str(out / "classifier.pkl"),
        "feat": str(out / "feature_config.json"),     # simpan senarai feature semasa training
        "stats": str(out / "training_stats.json"),    # optional: percentiles, target meta
    }

def _mtime(path: str) -> float:
    try: return os.path.getmtime(path)
    except: return 0.0

@st.cache_resource(show_spinner=False)
def _cached_load_models(reg_m: float, cls_m: float, feat_m: float, stats_m: float):
    """Cache berdasarkan mtime supaya auto reload bila notebook overwrite pkl/json."""
    p = model_paths()
    reg = joblib.load(p["reg"]) if os.path.exists(p["reg"]) else None
    cls = joblib.load(p["cls"]) if os.path.exists(p["cls"]) else None
    features = None
    if os.path.exists(p["feat"]):
        with open(p["feat"], "r", encoding="utf-8") as f:
            j = json.load(f)
            features = j.get("features") or j.get("feature_names")
    stats = None
    if os.path.exists(p["stats"]):
        with open(p["stats"], "r", encoding="utf-8") as f:
            stats = json.load(f)
    return {"reg": reg, "cls": cls, "features": features, "stats": stats}

def load_models() -> tuple[object|None, object|None, list[str]]:
    """Public loader: detect mtime & panggil cache."""
    p = model_paths()
    reg_m, cls_m, feat_m, stats_m = map(_mtime, [p["reg"], p["cls"], p["feat"], p["stats"]])
    bundle = _cached_load_models(reg_m, cls_m, feat_m, stats_m)
    reg, cls, features = bundle["reg"], bundle["cls"], bundle["features"]
    # Jika notebook tak simpan feature_config.json, fallback: guna numeric – target
    if features is None:
        df = load_cleaned_data()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        # buang target jika notebook simpan; jika tak pasti, biar semua numeric
        for tgt in ["total_crimes_reported"]:
            if tgt in num_cols:
                num_cols.remove(tgt)
        features = num_cols
    log().info(f"Models loaded. reg={reg is not None}, cls={cls is not None}, n_features={len(features)}")
    return bundle["reg"], bundle["cls"], features

def _align_features(X_user: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Selaraskan input ke susunan & set penuh features training. Isi yang hilang dengan median dataset."""
    df = load_cleaned_data()
    defaults = numeric_defaults(df)
    X = pd.DataFrame(columns=features)
    for f in features:
        if f in X_user.columns:
            X[f] = pd.to_numeric(X_user[f], errors="coerce")
        else:
            X[f] = defaults.get(f, 0.0)
    return X.fillna(0).astype(float)

def predict_realtime(X_input: pd.DataFrame) -> dict:
    """Ramalan real-time. Akan:
       - auto-load model terkini (ikut mtime)
       - align feature names
       - pulangkan regression intensity + class probabilities (Low/Med/High)
    """
    reg, cls, features = load_models()
    X = _align_features(X_input, features)

    out = {"intensity": None, "proba": {"Low": 0.33, "Medium": 0.34, "High": 0.33}}
    # Regression intensity
    if reg is not None:
        try:
            y = reg.predict(X)[0]
            out["intensity"] = float(y)
        except Exception as e:
            log().warning(f"Regression predict error: {e}")

    # Classification probs
    if cls is not None:
        try:
            probs = cls.predict_proba(X)[0]
            # anda boleh map indeks → label ikut notebook; di sini kita cuba 3 kelas
            # jika binary, buat Low/High dan bina Medium secara heuristik
            if len(probs) == 3:
                out["proba"] = {"Low": float(probs[0]), "Medium": float(probs[1]), "High": float(probs[2])}
            elif len(probs) == 2:
                out["proba"] = {"Low": float(probs[0]), "Medium": 0.0, "High": float(probs[1])}
            else:
                # fallback normalisasi
                probs = np.array(probs, dtype=float)
                probs = probs / probs.sum() if probs.sum() > 0 else np.array([1/3,1/3,1/3])
                out["proba"] = {"Low": float(probs[0]), "Medium": float(probs[1] if probs.size>1 else 0), "High": float(probs[-1])}
        except Exception as e:
            log().warning(f"Classification predict error: {e}")

    # Jika tiada regressor, hasilkan intensity dari expectation probs (skor 0..100 skala ringkas)
    if out["intensity"] is None:
        p = out["proba"]
        out["intensity"] = float(100*(0.2*p["Low"] + 0.5*p["Medium"] + 0.8*p["High"]))

    return out
