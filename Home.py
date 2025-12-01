# ================================================================
# 🏠 Home.py — CrimeLens India Dashboard (With Proposal Summary)
# CyberGlass Aqua Theme — Centered Title + Compact Layout
# ================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import zipfile
from pathlib import Path

from utils.common import theme_css, status_badge
from utils.guidance import guide_box, steps_box
from utils.intro_animation import show_intro
from utils.navbar import cyber_navbar
from utils.glow_panels import add_glow_effect

# ---------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------
st.set_page_config(page_title="CrimeLens India Dashboard", page_icon="🧊", layout="wide")
st.markdown(theme_css(), unsafe_allow_html=True)

show_intro()
cyber_navbar("Home", user_name="Muhammad Ilaf", user_role="Admin")
add_glow_effect()

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.markdown("""
<h1 style='text-align:center;font-size:52px;font-weight:900;color:#00e0ff;
text-shadow:0 0 25px #00b4d8;margin-bottom:-5px;'>
🧊 CrimeLens India Dashboard
</h1>

<p style='text-align:center;color:#caf0f8;font-size:19px;margin-top:0;'>
A machine-learning powered platform for crime analysis, district profiling, and risk intelligence.
</p>

<div class='cy-divider'></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# PROJECT SUMMARY
# ---------------------------------------------------------------
st.markdown("""
<div style="
background:rgba(255,255,255,0.05);
border:1px solid rgba(0,255,255,0.18);
padding:20px;
border-radius:14px;
box-shadow:0 0 20px rgba(0,200,255,0.18);
">
<h3 style='color:#90e0ef;margin-top:0;'>📘 Project Summary</h3>

<p style='color:#bde9ff;font-size:16px;'>
This project integrates <b>crime data</b>, <b>socio-economic indicators</b>, and 
<b>district-level development attributes</b> to understand how environment, 
economic pressure, and social conditions influence crime patterns in India.
</p>

<ul style='color:#bde9ff;font-size:15px;'>
<li>Identifies <b>crime hotspots</b> using district-level aggregation and trend analysis.</li>
<li>Studies relationships between <b>education, income, employment</b> and crime.</li>
<li>Uses <b>machine learning</b> to classify districts into Low/Medium/High crime levels.</li>
<li>Provides <b>explainability</b> for transparent model insights.</li>
</ul>

<p style='color:#bde9ff;font-size:15px;'>
The dashboard is designed for analysts, policymakers, researchers, and students 
who need clear, explainable insights—not just raw statistics.
</p>
</div>

<div class='cy-divider'></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# HOW TO USE
# ---------------------------------------------------------------
steps_box(
    "How to use this dashboard",
    [
        "1️⃣ Overview — Check KPIs & yearly patterns",
        "2️⃣ Analytics — Explore correlations & socioeconomic factors",
        "3️⃣ Map — Visualize district hotspots",
        "4️⃣ Prediction — ML-based forecasting",
        "5️⃣ Intelligence — AI-assisted summaries & detection",
        "6️⃣ Explainability — SHAP model insights",
        "7️⃣ Export — Download processed data"
    ],
    icon="🧭",
    open_default=False
)

# ---------------------------------------------------------------
# ZIP DATA LOADER (FINAL FIXED VERSION)
# ---------------------------------------------------------------
ZIP_PATH = Path("output/final_cleaned_crime_socioeconomic_data.zip")
TARGET_CSV = "final_cleaned_crime_socioeconomic_data.csv"

def load_dataset_from_zip():
    if not ZIP_PATH.exists():
        return None, f"ZIP file tidak dijumpai di: {ZIP_PATH}"

    try:
        with zipfile.ZipFile(ZIP_PATH) as z:
            csv_path = None

            # Cari CSV dalam apa-apa folder dalam ZIP
            for name in z.namelist():
                if name.endswith(TARGET_CSV):
                    csv_path = name
                    break

            if csv_path is None:
                return None, f"CSV '{TARGET_CSV}' tiada dalam ZIP. ZIP mengandungi: {z.namelist()}"

            with z.open(csv_path) as f:
                df = pd.read_csv(f, low_memory=False)
                return df, None

    except Exception as e:
        return None, f"Gagal membaca ZIP: {e}"

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
df, error = load_dataset_from_zip()

if error:
    status_badge("❌ Dataset Missing", "err")
    st.error(f"Unable to load dataset: {error}")
    st.stop()

df.columns = df.columns.str.lower()
status_badge("Dataset Loaded Successfully", "ok")

# ---------------------------------------------------------------
# KPI SNAPSHOT
# ---------------------------------------------------------------
st.markdown("### 📊 Dataset Snapshot", unsafe_allow_html=True)

records = len(df)
states = df["state"].nunique() if "state" in df else 0
districts = df["district"].nunique() if "district" in df else 0

years_range = "N/A"
if "year" in df:
    df["year"] = df["year"].astype(str).str.replace(r"\\.\\d+", "", regex=True).astype(int)
    years_range = f"{int(df['year'].min())} – {int(df['year'].max())}"

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{records:,}")
c2.metric("States", states)
c3.metric("Districts", districts)
c4.metric("Years Covered", years_range)

# ---------------------------------------------------------------
# MAIN GRAPHS
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.markdown("### 📈 National Crime Patterns", unsafe_allow_html=True)

colA, colB = st.columns(2)

# ---- YEARLY TREND ----
with colA:
    if "year" in df:
        yearly_df = (
            df.groupby("year")
            .size().reset_index(name="count")
            .sort_values("year")
        )

        figA = px.line(
            yearly_df, x="year", y="count",
            markers=True,
            template="plotly_dark",
            color_discrete_sequence=["#00e0ff"],
            title="Yearly Crime Trend"
        )
        figA.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=0))
        st.plotly_chart(figA, use_container_width=True)

# ---- TOP DISTRICTS ----
with colB:
    region_col = "district" if "district" in df else "state"

    top10 = (
        df.groupby(region_col)
        .size().reset_index(name="Report Count")
        .sort_values("Report Count", ascending=False)
        .head(10)
    )

    figB = px.bar(
        top10,
        x=region_col,
        y="Report Count",
        color="Report Count",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        title="Top 10 Districts by Crime Volume"
    )
    figB.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=0))
    figB.update_xaxes(tickangle=-30)
    st.plotly_chart(figB, use_container_width=True)

# ---------------------------------------------------------------
# QUICK NAVIGATION
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.markdown("### 🧭 Quick Navigation", unsafe_allow_html=True)

b1, b2, b3 = st.columns(3)
b4, b5, b6 = st.columns(3)

with b1:
    if st.button("📊 Overview", use_container_width=True):
        st.switch_page("pages/1_Overview.py")

with b2:
    if st.button("📈 Analytics", use_container_width=True):
        st.switch_page("pages/2_Analytics.py")

with b3:
    if st.button("🗺️ Map", use_container_width=True):
        st.switch_page("pages/3_Map.py")

with b4:
    if st.button("🤖 Prediction", use_container_width=True):
        st.switch_page("pages/4_Prediction.py")

with b5:
    if st.button("🧠 Intelligence", use_container_width=True):
        st.switch_page("pages/5_Crime_Intelligence.py")

with b6:
    if st.button("🔍 Explainability", use_container_width=True):
        st.switch_page("pages/6_Explainability.py")

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("""
<div class='cy-divider'></div>
<div style='text-align:center;color:#9cd8f7;font-size:14px;margin-top:10px;'>
© 2025 CrimeLens India Dashboard — CyberGlass Aqua Theme
</div>
""", unsafe_allow_html=True)
