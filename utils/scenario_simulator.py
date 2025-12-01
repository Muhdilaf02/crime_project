# ================================================================
# utils/scenario_simulator.py — v15.0 Pro
# What-If Simulator (uses exported notebook models if available)
# ================================================================
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

def _load_model(path: Path):
    try:
        with open(path, "rb") as f: 
            return pickle.load(f)
    except Exception:
        return None

def _to_numeric(df):
    d2 = df.copy()
    for c in d2.columns:
        if not np.issubdtype(d2[c].dtype, np.number):
            d2[c] = pd.to_numeric(d2[c], errors="coerce")
    return d2.fillna(0.0)

def _align(X: pd.DataFrame, features: list[str] | None):
    if not features: 
        return _to_numeric(X)
    out = pd.DataFrame(columns=features)
    for c in features:
        out[c] = X[c] if c in X.columns else 0.0
    return _to_numeric(out)

def what_if_simulator_mode(df: pd.DataFrame, paths: dict):
    st.subheader("🧪 What-If Simulator")

    out_dir = paths["OUTPUT"]
    reg = _load_model(out_dir / "regressor.pkl")
    clf = _load_model(out_dir / "classifier.pkl")
    feats = None
    if (out_dir / "feature_config.json").exists():
        try:
            feats = json.loads((out_dir/"feature_config.json").read_text(encoding="utf-8")).get("features", [])
        except Exception:
            feats = None

    if reg is None and clf is None:
        st.info("Models not found in ../output/. Simulator will compare baseline metrics only.")
    # pick 6 most impactful numeric columns (variance proxy)
    nums = df.select_dtypes(include="number")
    top = nums.var(numeric_only=True).sort_values(ascending=False).head(6).index.tolist()

    st.caption("Adjust the sliders below (we’ll simulate two scenarios vs. baseline).")
    c1, c2, c3 = st.columns(3)
    p = {}
    for i, col in enumerate(top):
        default = float(np.nanmedian(df[col]))
        q = df[col].quantile([0.05, 0.95]).values
        mn, mx = float(q[0]), float(q[1])
        target = [c1, c2, c3][i % 3]
        p[col] = target.slider(col.replace("_"," ").title(), min_value=mn, max_value=mx, value=default, step=(mx-mn)/50, key=f"sim_{col}")

    X0 = pd.DataFrame([{c: float(np.nanmedian(df[c])) for c in top}])
    X1 = pd.DataFrame([p])

    X0a = _align(X0, feats)
    X1a = _align(X1, feats)

    # intensity deltas
    def _pred_int(Xa):
        if reg is not None:
            try: return float(reg.predict(Xa)[0])
            except Exception: pass
        # fallback: sum of selected inputs as a proxy
        return float(Xa.iloc[0].sum())

    y0 = _pred_int(X0a)
    y1 = _pred_int(X1a)

    fig = px.bar(
        pd.DataFrame({"Scenario": ["Baseline", "Adjusted"], "Intensity": [y0, y1]}),
        x="Scenario", y="Intensity", text="Intensity", template="plotly_dark",
        color="Scenario", color_discrete_map={"Baseline":"#00e0ff","Adjusted":"#ffd166"}
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=420, title="Predicted Intensity — Baseline vs Adjusted")
    st.plotly_chart(fig, use_container_width=True)

    if clf is not None:
        try:
            lbl0 = str(clf.predict(X0a)[0]).capitalize()
            lbl1 = str(clf.predict(X1a)[0]).capitalize()
        except Exception:
            lbl0 = lbl1 = "Medium"
    else:
        lbl0 = "Medium"; lbl1 = "Medium"

    st.markdown(f"**Risk (Baseline → Adjusted):** {lbl0} → **{lbl1}**")
