# ================================================================
# utils/geo_overlay.py — v15.0 Pro
# Geospatial overlay (states/districts) with prediction context
# ================================================================
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

from utils.common import status_badge

def _load_geojson(candidates):
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None

def _slug(s): 
    return "".join(ch if ch.isalnum() else " " for ch in str(s).lower()).split()

def _slug_join(s):
    return " ".join(_slug(s))

def geo_overlay_mode(df: pd.DataFrame, paths: dict, auto_type=None, auto_value=None, auto_region=None):
    st.subheader("🌍 Geospatial Overlay")

    metric_options = [c for c in df.select_dtypes(include=[np.number]).columns if c != "year"]
    if not metric_options:
        st.info("No numeric columns available to map.")
        return

    metric = st.selectbox("Choose indicator to map", metric_options, index=0, key="geo_metric")
    level  = st.radio("Geography", ["State", "District"], horizontal=True, key="geo_level")

    # Load geojsons
    gj_d = _load_geojson([paths["DATA"]/ "districts.geojson", paths["ASSETS"]/ "districts.geojson"])
    gj_s = _load_geojson([paths["DATA"]/ "states.geojson",    paths["ASSETS"]/ "states.geojson"])

    if level == "District" and not gj_d:
        status_badge("❌ districts.geojson not found — showing fallback bar chart", "err")
        agg = df.groupby("district")[metric].mean().sort_values(ascending=False).head(20).reset_index()
        fig = px.bar(agg, x="district", y=metric, template="plotly_dark", title=f"Top districts by {metric}")
        st.plotly_chart(fig, use_container_width=True)
        return
    if level == "State" and not (gj_s or gj_d):
        status_badge("❌ states.geojson not found — need at least districts.geojson for fallback", "err")
        return

    # aggregate per region
    if level == "District":
        if "district" not in df.columns:
            st.error("❌ 'district' column missing.")
            return
        agg = df.groupby("district")[metric].mean().reset_index()
        agg["slug"] = agg["district"].astype(str).map(_slug_join)
        feature_key = "properties.name_slug"
        # ensure slugs in geojson
        for f in gj_d["features"]:
            props = f.setdefault("properties", {})
            dist = props.get("DISTRICT") or props.get("district") or props.get("NAME_2")
            props["name_slug"]  = _slug_join(dist) if dist else None
        fig = px.choropleth(
            agg, geojson=gj_d, locations="slug", color=metric,
            featureidkey=feature_key, color_continuous_scale="teal",
            template="plotly_dark", title=f"District overlay — {metric}"
        )
    else:
        # State-level: prefer true states geojson; fallback to district polygons colored by state agg
        state_col = "state" if "state" in df.columns else ("state_name" if "state_name" in df.columns else None)
        if state_col is None:
            st.error("❌ No 'state'/'state_name' column available.")
            return
        agg = df.groupby(state_col)[metric].mean().reset_index()
        agg["slug"] = agg[state_col].astype(str).map(_slug_join)

        if gj_s:
            for f in gj_s["features"]:
                props = f.setdefault("properties", {})
                name = props.get("STATE") or props.get("state") or props.get("NAME_1")
                props["state_slug"] = _slug_join(name) if name else None
            fig = px.choropleth(
                agg, geojson=gj_s, locations="slug", color=metric,
                featureidkey="properties.state_slug", color_continuous_scale="teal",
                template="plotly_dark", title=f"State overlay — {metric}"
            )
        else:
            # fallback via districts
            for f in gj_d["features"]:
                props = f.setdefault("properties", {})
                stn = props.get("STATE") or props.get("state") or props.get("ST_NM") or props.get("NAME_1")
                props["state_slug"] = _slug_join(stn) if stn else None
            # map each district to state's mean
            states_map = {r["slug"]: r[metric] for _, r in agg.iterrows()}
            # build frame for districts
            rows = []
            for f in gj_d["features"]:
                sslug = f["properties"].get("state_slug")
                if sslug in states_map:
                    rows.append({"name_slug": f["properties"].get("name_slug"), metric: states_map[sslug]})
            ddf = pd.DataFrame(rows).dropna()
            fig = px.choropleth(
                ddf, geojson=gj_d, locations="name_slug", color=metric,
                featureidkey="properties.name_slug", color_continuous_scale="teal",
                template="plotly_dark", title=f"State overlay (district polygons) — {metric}"
            )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # prediction context tag
    if auto_type or auto_value:
        st.caption(f"📎 Prediction context: {str(auto_type).title() if auto_type else '—'} = {auto_value if auto_value is not None else '—'}"
                   + (f" • Region: {auto_region}" if auto_region else ""))
