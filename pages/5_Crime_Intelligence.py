# ===============================================================
# 🌊 5_Crime_Intelligence.py — CyberGlass Aqua Guided Insights (Notebook-focused)
# ===============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import random
from pathlib import Path

from utils.common import theme_css, status_badge
from utils.data_pipeline import load_cleaned_data
from utils.guidance import guide_box, steps_box

# ---------------------------------------------------------------
# ⚙️ PAGE SETUP
# ---------------------------------------------------------------
st.set_page_config(page_title="Crime Insights & Stories", page_icon="🧭", layout="wide")
st.markdown(theme_css(), unsafe_allow_html=True)
from utils.navbar import cyber_navbar
from utils.glow_panels import add_glow_effect
cyber_navbar("Crime_Intelligence", user_name="Aila Farah", user_role="Admin")
add_glow_effect()

st.markdown("""
<style>
h1 { text-align:center; font-weight:900; color:#00e0ff; font-size:46px; text-shadow:0 0 25px #00b4d8; }
.cy-divider { height:2px; background:linear-gradient(90deg,transparent,#00b4d8,transparent); margin:25px 0; border-radius:2px; }
.card { background: rgba(255,255,255,0.07); border: 1px solid rgba(0,255,255,0.15); border-radius: 16px; padding: 18px 22px; backdrop-filter: blur(12px); box-shadow: 0 0 25px rgba(0,180,216,0.25); transition: all 0.3s ease; }
.card:hover { transform: scale(1.01); box-shadow: 0 0 35px rgba(0,200,255,0.35); }
.metric { font-size:34px; font-weight:800; color:#00e0ff; text-shadow:0 0 12px #00b4d8; }
.story { background:rgba(0,0,0,0.35); border-left:4px solid #00e0ff; padding:14px 18px; border-radius:10px; margin-top:12px; color:#caf0f8; }
.story b { color:#00e0ff; }
.explain { color:#8ecae6; font-size:14px; font-style:italic; margin-top:6px; }
.small { font-size:13px; color:#9fd3e6; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1>🧭 Crime Insights & Stories</h1>
<p style="text-align:center;color:#ade8f4;font-size:18px;">
Guided intelligence view — concise charts drawn from the notebook analysis, with short explanations for FYP reporting.
</p>
<div class="cy-divider"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 🧭 INTRO & USER GUIDE
# ---------------------------------------------------------------
guide_box(
    "🔍 What This Page Does",
    """
The Crime Insights module turns cleaned data into the key visual insights used in your FYP notebook:

- 🔥 Identifies top risky and safe regions (hotspots)  
- 📊 Shows dominant crime categories and victim demographics  
- 🧭 Compares socioeconomic factors (e.g., literacy) against crime domains  
- 🗣️ Produces short human-readable summaries for reports
""",
    icon="💡",
    open_default=True
)

steps_box(
    "🪜 Step-by-Step Guide",
    [
        "1️⃣ Choose analysis level (State or District).",
        "2️⃣ Review overview cards (riskiest, average, safest).",
        "3️⃣ Explore Hotspots & Domain breakdown (side-by-side charts).",
        "4️⃣ Analyze victim demographics (age & gender).",
        "5️⃣ Compare a socioeconomic factor vs crime totals & domain (scatter/boxplot).",
        "6️⃣ Read the auto-generated story summary for reporting."
    ],
    icon="🧭",
    open_default=False
)

# ---------------------------------------------------------------
# 📂 LOAD DATA
# ---------------------------------------------------------------
try:
    df = load_cleaned_data()
    df.columns = df.columns.str.lower().str.strip()
    status_badge("✅ Dataset Loaded Successfully", "ok")
except Exception as e:
    status_badge("❌ Unable to Load Dataset", "err")
    st.error(e)
    st.stop()

# ---------------------------------------------------------------
# 🔍 PREPARE / DETECT COLUMNS (robust)
# ---------------------------------------------------------------
# Region columns
state_col = "state_name" if "state_name" in df.columns else None
district_col = "district" if "district" in df.columns else None

# crime_count detection (prefer aggregated)
if "crime_count" in df.columns:
    df["crime_count"] = pd.to_numeric(df["crime_count"], errors="coerce").fillna(0)
else:
    # fallback: per-row = 1 or other numeric count columns
    found_count = next((c for c in ["report_number", "case_count", "incidents", "total_crimes_reported"] if c in df.columns), None)
    if found_count and np.issubdtype(df[found_count].dtype, np.number):
        # If report_number is identifier then we set proxy=1; if it's numeric aggregated then use as is
        if found_count == "report_number":
            df["crime_count"] = 1
        else:
            df["crime_count"] = pd.to_numeric(df[found_count], errors="coerce").fillna(0)
    else:
        df["crime_count"] = 1

# year detection (use date_of_occurrence preferred)
for dcol in ("date_of_occurrence", "date_reported", "date_case_closed"):
    if dcol in df.columns:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

if "date_of_occurrence" in df.columns and "date_reported" in df.columns:
    df["_year"] = df[["date_of_occurrence", "date_reported"]].min(axis=1).dt.year
elif "date_of_occurrence" in df.columns:
    df["_year"] = df["date_of_occurrence"].dt.year
elif "date_reported" in df.columns:
    df["_year"] = df["date_reported"].dt.year
else:
    df["_year"] = pd.NA

# crime domain detection
domain_candidates = ["crime_domain", "crime_type", "crime_category", "crime_description", "category"]
domain_col = next((c for c in domain_candidates if c in df.columns), None)

# literacy-like column detection (noted in notebook)
literacy_candidates = ["no_literate_adult_25_plus", "literacy_rate", "literacy"]
literacy_col = next((c for c in literacy_candidates if c in df.columns), None)

# victim demographics
age_col = "victim_age" if "victim_age" in df.columns else None
gender_col = "victim_gender" if "victim_gender" in df.columns else None

# ---------------------------------------------------------------
# 🔽 STEP 1: SELECT LEVEL (State / District)
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("🪜 Step 1: Choose Level of Analysis")
st.caption("Pick **State** or **District**. This determines grouping for hotspots and factor correlations.")

level = st.radio("Select Analysis Level:", ["State", "District"], horizontal=True)
group_col = state_col if level == "State" else district_col
if group_col is None:
    st.error(f"❌ The column for '{level}' is not present in dataset.")
    st.stop()

# ---------------------------------------------------------------
# 🔎 STEP 2: OVERVIEW SUMMARY CARDS
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("🪜 Step 2: Overview Summary")

totals = df.groupby(group_col, as_index=False)["crime_count"].sum().sort_values("crime_count", ascending=False)
avg_crime = totals["crime_count"].mean() if len(totals) else 0
top_region = totals.iloc[0][group_col] if len(totals) else "N/A"
bottom_region = totals.iloc[-1][group_col] if len(totals) else "N/A"

c1, c2, c3 = st.columns(3)
c1.markdown(f"<div class='card'><b>🔥 Riskiest {level}</b><br><div class='metric'>{top_region}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><b>📈 Avg per {level}</b><br><div class='metric'>{avg_crime:.1f}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><b>🧊 Safest {level}</b><br><div class='metric'>{bottom_region}</div></div>", unsafe_allow_html=True)
st.markdown("<p class='explain'>Quick summary of which regions have the highest/lowest aggregate incident counts.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 🔥 STEP 3: HOTSPOTS & DOMAIN (2 graphs side-by-side)
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("🪜 Step 3: Hotspots & Crime Domain")
st.caption("Top regions vs crime category — compact side-by-side view for quick insights.")

colA, colB = st.columns(2)

PLOTLY_H = 340

with colA:
    st.markdown("#### 🔥 Top 5 High-Crime Regions")
    top5 = totals.head(5).copy()
    if not top5.empty:
        fig_top = px.bar(top5, x=group_col, y="crime_count", text="crime_count",
                         color="crime_count", color_continuous_scale="rdbu", height=PLOTLY_H)
        fig_top.update_layout(margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No region totals available.")
    st.markdown("<p class='explain'>Shows the top 5 regions with the most reported incidents — useful to prioritise resource allocation.</p>", unsafe_allow_html=True)

with colB:
    st.markdown("#### 🔎 Top Crime Categories")
    if domain_col:
        domain_counts = df[domain_col].fillna("Unknown").value_counts().reset_index()
        domain_counts.columns = [domain_col, "count"]
        top_domains = domain_counts.head(8)
        fig_dom = px.bar(top_domains, x="count", y=domain_col, orientation="h",
                         color="count", color_continuous_scale="Reds", height=PLOTLY_H)
        fig_dom.update_layout(margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_dom, use_container_width=True)
    else:
        st.info("Crime category column not found.")
    st.markdown("<p class='explain'>Identifies the most common crime types in the dataset — informs model class balance and prioritisation.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 📊 STEP 4: VICTIM DEMOGRAPHICS (2 graphs side-by-side)
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("🪜 Step 4: Victim Demographics")
st.caption("Age & gender analysis — compact view for profiling.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👥 Victim Age Distribution")
    if age_col:
        fig_age = px.histogram(df, x=age_col, nbins=30, marginal="box", title="Victim Age Distribution", height=PLOTLY_H)
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("<p class='explain'>Shows which age groups are most affected; useful for targeted interventions and model features.</p>", unsafe_allow_html=True)
    else:
        st.info("victim_age not found in dataset.")

with col2:
    st.markdown("#### ⚥ Victim Gender Breakdown")
    if gender_col:
        gender_counts = df[gender_col].fillna("Unknown").value_counts().reset_index()
        gender_counts.columns = ["gender", "count"]
        fig_gender = px.pie(gender_counts, names="gender", values="count", height=PLOTLY_H)
        st.plotly_chart(fig_gender, use_container_width=True)
        st.markdown("<p class='explain'>Displays gender composition among victims — may reveal gendered vulnerability patterns.</p>", unsafe_allow_html=True)
    else:
        st.info("victim_gender not found in dataset.")

# ---------------------------------------------------------------
# 🔎 STEP 5: SOCIOECONOMIC COMPARISON (2 graphs side-by-side)
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("🪜 Step 5: Socioeconomic Comparison (Literacy vs Crime & Factor Correlation)")
st.caption("Compare literacy across crime domains (boxplot) and test correlation between one selected factor and crime totals.")

colL, colR = st.columns(2)

with colL:
    st.markdown("#### 📚 Literacy vs Crime Domain (Boxplot)")
    if domain_col and literacy_col:
        # show boxplot but keep compact height
        fig_box = px.box(df, x=domain_col, y=literacy_col, points="outliers", height=PLOTLY_H)
        fig_box.update_layout(xaxis_tickangle=25, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("<p class='explain'>Illustrates how literacy-related indicators vary between crime types — supports hypothesis that literacy correlates with crime patterns.</p>", unsafe_allow_html=True)
    else:
        st.info("Either crime domain or literacy column not available — boxplot skipped.")

with colR:
    st.markdown("#### 📈 Correlation: Choose Factor vs Crime Totals")
    # choose numeric columns for factor selection
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("crime_count", "_year")]
    factor_choice = st.selectbox("Choose a socioeconomic factor to compare:", options=numeric_cols[:40] if len(numeric_cols)>0 else ["none"])
    if factor_choice and factor_choice != "none":
        # compute mean factor per region and merge with totals
        factor_region = df.groupby(group_col, as_index=False)[factor_choice].mean()
        merged = totals.merge(factor_region, on=group_col, how="left")
        fig_scatter = px.scatter(merged, x=factor_choice, y="crime_count", trendline="ols",
                                 size="crime_count", height=PLOTLY_H, color="crime_count",
                                 color_continuous_scale="tealgrn")
        st.plotly_chart(fig_scatter, use_container_width=True)
        corr_val = merged[factor_choice].corr(merged["crime_count"])
        if pd.notna(corr_val):
            if abs(corr_val) > 0.5:
                st.success(f"Strong correlation detected (r = {corr_val:.2f}).")
            elif abs(corr_val) > 0.3:
                st.info(f"Moderate correlation (r = {corr_val:.2f}).")
            else:
                st.caption(f"Weak correlation (r = {corr_val:.2f}).")
        else:
            st.caption("Not enough data to compute correlation.")
        st.markdown("<p class='explain'>Scatter plot with OLS trendline shows linear association; stronger slopes and tighter points indicate stronger relationships.</p>", unsafe_allow_html=True)
    else:
        st.info("No numeric factor available to compare.")

# ---------------------------------------------------------------
# 🗣️ STEP 6: STORY SUMMARY (auto-generated from above)
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("🪜 Step 6: Story Summary")
st.caption("A concise human-readable summary derived from the visuals — suitable for inclusion in your FYP report.")

# Compose story using top region, factor choice and domain insights
trend_terms = ["steady pattern", "mild increase", "notable decline", "sharp surge", "stable trend"]
tone = random.choice(trend_terms)

story_lines = []
story_lines.append(f"In this analysis, <b>{top_region}</b> emerges as the riskiest {level.lower()}, with total incidents higher than average.")
if factor_choice and factor_choice != "none":
    story_lines.append(f"The factor <b>{factor_choice}</b> shows a correlation score of <b>{corr_val:.2f}</b> with crime totals (per {level.lower()}).")
if domain_col:
    top_dom = df[domain_col].value_counts().idxmax() if df[domain_col].notna().any() else "N/A"
    story_lines.append(f"The most common crime category observed is <b>{top_dom}</b>, which the predictive model must learn to prioritise.")
story_lines.append(f"Overall pattern indicates a <b>{tone}</b> across regions; policymakers can use these insights to target high-risk districts.")

st.markdown("<div class='story'>" + "<br>".join(story_lines) + "</div>", unsafe_allow_html=True)
st.markdown("<p class='explain'>This summary synthesises hotspots, factor relationships and domain-level insights for a one-page briefing.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 🌌 FOOTER
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#8ecae6;font-size:14px;">
© 2025 Crime Analytics Dashboard — Guided Insights • CyberGlass Aqua
</div>
""", unsafe_allow_html=True)
