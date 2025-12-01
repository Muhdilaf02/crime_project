# ================================================================
# 🗺️ 3_Map.py — Crime Hotspot Map (FINAL FIXED VERSION)
# - Single fast basemap
# - District-only mode (state removed)
# - Correct GeoJSON path
# - No Stamen error
# - Map & chart same size, aligned
# - Interpretation guide above map
# ================================================================

import json
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import folium
from folium.plugins import HeatMap
from folium import CircleMarker
import branca.colormap as cm
import plotly.express as px
from shapely.geometry import shape
from streamlit.components.v1 import html as st_html

# Optional project modules
from utils.common import theme_css, status_badge
from utils.guidance import guide_box, steps_box
from utils.navbar import cyber_navbar
from utils.glow_panels import add_glow_effect

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(page_title="Crime Hotspot Map", page_icon="🗺️", layout="wide")
st.markdown(theme_css(), unsafe_allow_html=True)
cyber_navbar("Map", user_name="Aila Farah", user_role="Admin")
add_glow_effect()

st.markdown("""
<h1 style="text-align:center;color:#00e0ff;text-shadow:0 0 20px #00b4d8;">
🗺️ Crime Hotspot Intelligence Map
</h1>
<p style="text-align:center;color:#ade8f4;">
District-level heatmap + hotspot circle markers
</p>
<div class="cy-divider"></div>
""", unsafe_allow_html=True)

# ================================================================
# DATASET & GEOJSON PATHS
# ================================================================
CSV_PATH = Path("C:/Users/ilafl/projects/crime_project/output/final_cleaned_crime_socioeconomic_data.csv")
GEOJSON_PATH = Path("C:/Users/ilafl/projects/crime_project/output/districts.geojson")

if not CSV_PATH.exists():
    st.error(f"❌ CSV not found: {CSV_PATH}")
    st.stop()
if not GEOJSON_PATH.exists():
    st.error(f"❌ GeoJSON not found: {GEOJSON_PATH}")
    st.stop()

status_badge("✅ Dataset Loaded", "ok")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.lower().str.strip()

with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# ================================================================
# ENSURE YEAR COLUMN EXISTS
# ================================================================
if "year" not in df.columns:
    date_cols = [c for c in df.columns if "date" in c]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        df["year"] = df[date_cols[0]].dt.year
    else:
        df["year"] = np.nan

# ================================================================
# GUIDE + STEPS
# ================================================================
guide_box(
    "About This Page",
    """
    This map visualizes **crime hotspots** using two layers:
    - 🔥 **Heatmap** — shows overall intensity of crime
    - 🟡 **Circle markers** — each district sized by crime volume

    This helps identify **critical hotspots**, **dense crime zones**,  
    and **regions needing policy attention**.
    """,
    icon="📘",
    open_default=True
)

steps_box(
    "How to Use This Page",
    [
        "1️⃣ Select a year to filter crime distribution",
        "2️⃣ View heatmap for overall intensity patterns",
        "3️⃣ Hover hotspot circles to read district values",
        "4️⃣ Check the bar chart for exact top-hotspot ranking",
    ],
    icon="🧭",
    open_default=False
)

# ================================================================
# FILTERS
# ================================================================
years_available = sorted(df["year"].dropna().unique().tolist())
year_choice = st.selectbox("Filter by Year:", ["All"] + years_available)

df_filtered = df.copy()
if year_choice != "All":
    df_filtered = df[df["year"] == int(year_choice)]

# ================================================================
# CRIME METRIC
# ================================================================
if "total_crimes_reported" in df_filtered.columns:
    df_filtered["metric"] = pd.to_numeric(df_filtered["total_crimes_reported"], errors="coerce").fillna(0)
else:
    df_filtered["metric"] = 1

# District name formatting
df_filtered["slug"] = df_filtered["district"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True)

# ================================================================
# GEOJSON CENTROIDS
# ================================================================
from shapely.geometry import shape

centroids = []
for feature in geojson_data["features"]:
    props = feature["properties"]
    district_name = props.get("district") or props.get("DISTRICT") or props.get("NAME_2")
    if district_name:
        slug = district_name.lower().replace(" ", "")
        geom = shape(feature["geometry"])
        centroids.append({
            "slug": slug,
            "raw_name": district_name,
            "lat": geom.centroid.y,
            "lon": geom.centroid.x
        })

centroid_df = pd.DataFrame(centroids)

# Merge crime data with geo coordinates
df_merged = df_filtered.groupby("slug", as_index=False)["metric"].sum()
df_merged["slug"] = df_merged["slug"].str.replace(" ", "")
final_df = df_merged.merge(centroid_df, on="slug", how="left").dropna()

# ================================================================
# INTERPRETATION GUIDE ABOVE MAP
# ================================================================
st.markdown("""
<div style="background:rgba(255,255,255,0.05);padding:15px;border-radius:12px;border:1px solid rgba(0,255,255,0.2);margin-bottom:15px;">
<b>📘 Interpretation Guide</b><br>
• Red/yellow zones = high crime intensity (hotspots)<br>
• Blue/green zones = lower reported activity<br>
• Larger circles = higher crime volume<br>
• Hover markers to see district names and values<br>
• Use the bar chart to compare exact ranking
</div>
""", unsafe_allow_html=True)

# ================================================================
# BUILD MAP
# ================================================================
center_lat = final_df["lat"].mean()
center_lon = final_df["lon"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="cartodbpositron",   # FAST + RELIABLE + CLEAN
    attr="CartoDB"
)

# Color scale
cmin, cmax = final_df["metric"].min(), final_df["metric"].max()
colormap = cm.LinearColormap(["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"], vmin=cmin, vmax=cmax)

# Heatmap
HeatMap(
    final_df[["lat", "lon", "metric"]].values,
    radius=18, blur=15, max_zoom=1
).add_to(m)

# Circle markers
for _, row in final_df.iterrows():
    CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5 + (row["metric"] / cmax) * 15,
        fill=True,
        fill_color=colormap(row["metric"]),
        fill_opacity=0.9,
        color="white",
        weight=1,
        tooltip=f"{row['raw_name']} — {row['metric']:.0f}"
    ).add_to(m)

# Display map
col1, col2 = st.columns([1, 1])

with col1:
    st_html(m._repr_html_(), height=600)

# ================================================================
# HOTSPOT BAR CHART (same size)
# ================================================================
top_df = final_df.sort_values("metric", ascending=False).head(15)

fig = px.bar(
    top_df[::-1],
    x="metric",
    y="raw_name",
    orientation="h",
    template="plotly_dark",
    color="metric",
    color_continuous_scale="Turbo",
    title="Top Hotspot Districts"
)
fig.update_layout(height=600)

with col2:
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# FOOTER
# ================================================================
st.markdown("""
<div class="cy-divider"></div>
<div style="text-align:center;color:#8ecae6;">
© 2025 Crime Analytics Dashboard — Hotspot Mapping Module
</div>
""", unsafe_allow_html=True)
