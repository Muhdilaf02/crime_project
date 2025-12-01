# ================================================================
# 📊 1_Overview.py — CyberGlass Aqua Edition (Compact, Rewritten)
# - Compact layout (no-scroll friendly)
# - 2-graph side-by-side sections
# - Explanations for each graph for FYP use
# ================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# project utilities (assumes these exist in your repo as before)
from utils.common import theme_css, status_badge
from utils.data_pipeline import load_cleaned_data
from utils.guidance import guide_box, steps_box
from utils.navbar import cyber_navbar
from utils.glow_panels import add_glow_effect

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
st.markdown(theme_css(), unsafe_allow_html=True)
cyber_navbar("Overview", user_name="Aila Farah", user_role="Admin")
add_glow_effect()

# Small CSS tweaks for compact look
st.markdown("""
<style>
/* Compact metric boxes */
.metric-box { background: rgba(255,255,255,0.02); border-radius:10px; padding:10px; text-align:center; }
.metric-value { font-size:20px; font-weight:800; color:#00e0ff; }
.metric-label { font-size:12px; color:#bfefff; margin-top:6px; }
.cy-divider { height:2px; background:linear-gradient(90deg,transparent,#00b4d8,transparent); margin:14px 0; border-radius:2px; }
.card-compact { background: rgba(255,255,255,0.03); padding:10px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# -------- HEADER & DESCRIPTION --------
st.markdown("""
<h1 style='text-align:center;font-size:40px;color:#00e0ff;text-shadow:0 0 18px #00b4d8;'>📊 Overview Of Crime</h1>
<p style='text-align:center;color:#bfefff;font-size:14px;margin-top:-8px;'>
Compact summary of national records, trends, hotspots and key socio-economic patterns — ready for FYP reporting.
</p>
<div class='cy-divider'></div>
""", unsafe_allow_html=True)

# -------- GUIDE & STEPS --------
guide_box(
    "About this page",
    """
This Overview page presents a compact set of visuals used in the notebook analysis:
- Key KPIs (records, states, districts, years)
- Yearly trend + Top regions (side-by-side)
- Crime domain & victim demographics (side-by-side)
- Quick socio-economic comparison (selectable factor)
Each chart includes a short explanation you can copy into your FYP.
""",
    icon="📘",
    open_default=True
)

steps_box(
    "How to read this page",
    [
        "1. Check the KPI cards for dataset coverage.",
        "2. Study Yearly Trend (left) and Top Regions (right).",
        "3. Explore Crime Domain vs Victim demographics.",
        "4. Use the factor selector to test one socio-economic relationship."
    ],
    icon="🧭",
    open_default=False
)

# -------- LOAD DATA (robust) --------
try:
    df = load_cleaned_data()
    # normalize column names
    df.columns = df.columns.str.lower().str.strip()
    status_badge("✅ Dataset loaded", "ok")
except Exception as e:
    status_badge("❌ Failed to load data", "err")
    st.error(f"Error loading dataset: {e}")
    st.stop()

if df.empty:
    st.error("Dataset is empty.")
    st.stop()

# -------- DETECT COLUMNS (robust, fallbacks) --------
# region columns
state_col = next((c for c in ["state_name", "state", "admin1", "province"] if c in df.columns), None)
district_col = next((c for c in ["district", "district_name", "admin2"] if c in df.columns), None)

# crime count
if "crime_count" in df.columns and pd.api.types.is_numeric_dtype(df["crime_count"]):
    df["crime_count"] = pd.to_numeric(df["crime_count"], errors="coerce").fillna(0)
else:
    # fallback: use per-row count proxy
    df["crime_count"] = 1

# year detection
date_candidates = ["_year", "year", "date_of_occurrence", "date_reported"]
year_col = None
if "_year" in df.columns:
    year_col = "_year"
elif "year" in df.columns:
    year_col = "year"
else:
    # try to parse date columns
    for d in ["date_of_occurrence", "date_reported", "date_case_closed"]:
        if d in df.columns:
            df[d] = pd.to_datetime(df[d], errors="coerce")
            df["_year"] = df[d].dt.year
            year_col = "_year"
            break

# domain detection
domain_col = next((c for c in ["crime_domain", "crime_type", "crime_category", "category", "crime_description"] if c in df.columns), None)

# victim demographics
age_col = "victim_age" if "victim_age" in df.columns else None
gender_col = "victim_gender" if "victim_gender" in df.columns else None

# choose group column for region analysis
group_col = district_col if district_col else state_col

# -------- KPI CARDS (compact) --------
st.subheader("📌 Key Metrics")
records = len(df)
states_count = df[state_col].nunique() if state_col else 0
districts_count = df[district_col].nunique() if district_col else 0
years_range = "N/A"
if year_col and df[year_col].notna().any():
    years = df[year_col].dropna().astype(int)
    years_range = f"{int(years.min())} – {int(years.max())}"

kcol1, kcol2, kcol3, kcol4 = st.columns([1,1,1,1])
kcol1.markdown(f"<div class='metric-box'><div class='metric-value'>{records:,}</div><div class='metric-label'>🧾 Total Reports</div></div>", unsafe_allow_html=True)
kcol2.markdown(f"<div class='metric-box'><div class='metric-value'>{states_count:,}</div><div class='metric-label'>🏙️ Total States</div></div>", unsafe_allow_html=True)
kcol3.markdown(f"<div class='metric-box'><div class='metric-value'>{districts_count:,}</div><div class='metric-label'>📍 Districts</div></div>", unsafe_allow_html=True)
kcol4.markdown(f"<div class='metric-box'><div class='metric-value'>{years_range}</div><div class='metric-label'>📅 Years Covered</div></div>", unsafe_allow_html=True)

st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)

# -------- SECTION: Yearly Trend (left) & Top Regions (right) --------
st.subheader("📈 Trends & Hotspots")
st.caption("Yearly volume and top regions — side-by-side for quick comparison.")

h = 330  # compact height for no-scroll

col_l, col_r = st.columns(2)

with col_l:
    # Yearly trend
    if year_col and df[year_col].notna().any():
        # ensure integer year
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
        trend = df.groupby(year_col)["crime_count"].sum().reset_index().sort_values(year_col)
        fig_trend = px.line(trend, x=year_col, y="crime_count", markers=True,
                            title="Reports Over Years", template="plotly_dark", height=h,
                            labels={year_col: "Year", "crime_count": "Reports"})
        fig_trend.update_traces(line=dict(color="#00e0ff", width=3))
        fig_trend.update_layout(margin=dict(l=10, r=10, t=36, b=10))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("<p class='small'>Line chart shows how the number of reports changes year-to-year. Use to spot surges or declines.</p>", unsafe_allow_html=True)
    else:
        st.info("No year data available to plot yearly trend.")

with col_r:
    if group_col:
        region_tot = df.groupby(group_col)["crime_count"].sum().reset_index().sort_values("crime_count", ascending=False).head(8)
        fig_regions = px.bar(region_tot, x=group_col, y="crime_count",
                             title=f"Top {group_col.title()}s by Reports", template="plotly_dark",
                             color="crime_count", color_continuous_scale="rdbu", height=h)
        fig_regions.update_layout(margin=dict(l=10, r=10, t=36, b=10), xaxis_tickangle=-20)
        st.plotly_chart(fig_regions, use_container_width=True)
        st.markdown("<p class='small'>Bar chart highlights the regions with most reports (hotspots). Useful for prioritising interventions.</p>", unsafe_allow_html=True)
    else:
        st.info("No region column found for hotspot analysis.")

st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)

# -------- SECTION: Crime Domain (left) & Victim Demographics (right) --------
st.subheader("📊 Crime Domain & Victim Profile")
st.caption("Important domain-level and victim demographic charts used in the notebook.")

col_a, col_b = st.columns(2)

with col_a:
    if domain_col:
        dom_counts = df[domain_col].fillna("Unknown").value_counts().reset_index().head(10)
        dom_counts.columns = [domain_col, "count"]
        fig_dom = px.bar(dom_counts, x="count", y=domain_col, orientation="h",
                         title="Top Crime Domains", template="plotly_dark", height=h,
                         color="count", color_continuous_scale="Reds")
        fig_dom.update_layout(margin=dict(l=10, r=10, t=36, b=10))
        st.plotly_chart(fig_dom, use_container_width=True)
        st.markdown("<p class='small'>Shows the most common categories of crime. This informs model labels and intervention types.</p>", unsafe_allow_html=True)
    else:
        st.info("No crime domain column detected to plot domain distribution.")

with col_b:
    # Two small subplots stacked compactly: age histogram and gender pie
    if age_col or gender_col:
        # age plot
        if age_col:
            # clean ages
            ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
            if not ages.empty:
                fig_age = px.histogram(ages, nbins=25, title="Victim Age Distribution", template="plotly_dark", height=int(h/2)-20)
                fig_age.update_layout(margin=dict(l=10, r=10, t=30, b=6))
                st.plotly_chart(fig_age, use_container_width=True)
                st.markdown("<p class='small'>Age histogram helps identify vulnerable age groups.</p>", unsafe_allow_html=True)
            else:
                st.info("Victim age column present but contains no numeric values.")
        else:
            st.info("Victim age column not available.")
        # gender plot
        if gender_col:
            gender_counts = df[gender_col].fillna("Unknown").value_counts().reset_index()
            gender_counts.columns = ["gender", "count"]
            fig_gender = px.pie(gender_counts, names="gender", values="count", title="Victim Gender", template="plotly_dark", height=int(h/2)-20)
            fig_gender.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_gender, use_container_width=True)
            st.markdown("<p class='small'>Pie chart shows proportion of victims by gender.</p>", unsafe_allow_html=True)
        else:
            st.info("Victim gender column not available.")
    else:
        st.info("No victim demographic columns detected (age/gender).")

st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)

# -------- SECTION: Socioeconomic Factor Comparison (compact) --------
st.subheader("🔎 Socioeconomic Comparison (Compact)")
st.caption("Select one socioeconomic factor to compare vs region crime totals (scatter with trendline).")

# Candidate numeric socioeconomic columns (limit to common ones found in your dataset)
socio_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("crime_count", "_year", "year")]
# short list for UI
options = socio_candidates[:40] if len(socio_candidates) > 0 else []
if options:
    factor = st.selectbox("Choose factor", options=options, index=0)
    # prepare merged df per region
    if group_col:
        factor_region = df.groupby(group_col, as_index=False)[factor].mean()
        merged = df.groupby(group_col, as_index=False)["crime_count"].sum().merge(factor_region, on=group_col)
        fig_fact = px.scatter(merged, x=factor, y="crime_count", size="crime_count",
                              trendline="ols", template="plotly_dark", height=360, color="crime_count", color_continuous_scale="tealgrn")
        fig_fact.update_layout(margin=dict(l=10, r=10, t=36, b=10))
        st.plotly_chart(fig_fact, use_container_width=True)
        corr_val = merged[factor].corr(merged["crime_count"])
        if pd.notna(corr_val):
            if abs(corr_val) > 0.5:
                st.success(f"Strong correlation (r = {corr_val:.2f})")
            elif abs(corr_val) > 0.3:
                st.info(f"Moderate correlation (r = {corr_val:.2f})")
            else:
                st.caption(f"Weak correlation (r = {corr_val:.2f})")
        st.markdown("<p class='small'>Scatter with OLS line — use to evaluate if this socioeconomic factor aligns with higher crime totals.</p>", unsafe_allow_html=True)
    else:
        st.info("Region grouping not available to compute factor vs crime.")
else:
    st.info("No numeric socio-economic columns detected to compare.")

st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)

# -------- STORY SUMMARY --------
st.subheader("🪄 Quick Story Summary")
# Build a short deterministic summary from above results
top_region = region_tot[group_col].iloc[0] if group_col and not region_tot.empty else "N/A"
top_domain = df[domain_col].mode().iloc[0] if domain_col and df[domain_col].notna().any() else "N/A"
summary_lines = [
    f"<b>{top_region}</b> emerges as a top hotspot in the dataset.",
    f"The most frequent crime category observed is <b>{top_domain}</b>.",
]
if options:
    summary_lines.append(f"The factor <b>{factor}</b> shows correlation r = {corr_val:.2f} with region crime totals.")
summary_lines.append("Use these points in your FYP to motivate why these regions and factors were selected for modelling and intervention.")
st.markdown("<div class='card-compact'>" + "<br>".join(summary_lines) + "</div>", unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#8ecae6;font-size:13px;'>© 2025 Crime Analytics Dashboard — Overview • CyberGlass Aqua</div>", unsafe_allow_html=True)
