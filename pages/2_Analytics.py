# ====================================================================== 
# 2_Analytics.py — Final (5 tabs, 2 graphs per tab, compact)
# ======================================================================
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json, joblib, warnings, zipfile
warnings.filterwarnings("ignore")

# UI helpers
from utils.common import theme_css
from utils.navbar import cyber_navbar
from utils.glow_panels import add_glow_effect


# ---------------- Page config & header ----------------
st.set_page_config(page_title="Unified Intelligence Studio — Crime Analytics",
                   page_icon="📊", layout="wide")
st.markdown(theme_css(), unsafe_allow_html=True)
cyber_navbar("Analytics", user_name="Aila Farah", user_role="Admin")
add_glow_effect()

st.markdown("""
<h1 style='text-align:center;font-size:44px;color:#00e0ff;text-shadow:0 0 18px #00b4d8;'>
Crime & Socioeconomic Analytics Dashboard
</h1>
<p style='text-align:center;color:#bde0fe;font-size:15px;line-height:1.4;max-width:900px;margin:0 auto;'>
This dashboard provides an analytical view of crime distribution, victim profiling, hotspot identification,
socioeconomic relationships and multivariate correlations — supporting the FYP objective 
<b>"Predictive Crime Analysis Based on Socioeconomic Indicators"</b>.
</p>
<div style="height:12px"></div>
""", unsafe_allow_html=True)

# ======================================================================
# ---------------- Load dataset from ZIP (FINAL FIXED VERSION) ---------
# ======================================================================

ZIP_PATH = Path("output/final_cleaned_crime_socioeconomic_data.zip")
CSV_NAME = "final_cleaned_crime_socioeconomic_data.csv"

if not ZIP_PATH.exists():
    st.error(f"ZIP file not found at: {ZIP_PATH}")
    st.stop()

try:
    with zipfile.ZipFile(ZIP_PATH) as z:
        if CSV_NAME not in z.namelist():
            st.error(f"CSV '{CSV_NAME}' not found in ZIP. ZIP contains: {z.namelist()}")
            st.stop()
        with z.open(CSV_NAME) as f:
            df = pd.read_csv(f, low_memory=False)
except Exception as e:
    st.error(f"Failed to load ZIP: {e}")
    st.stop()

# Normalize columns
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace(r"[^\w\d_]", "", regex=True)
)


# ======================================================================
# ---------------- Continue original code ------------------------------
# ======================================================================

# Parse date columns
for dcol in ("date_of_occurrence", "date_reported", "date_case_closed"):
    if dcol in df.columns:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

df["_year"] = pd.NA
if "date_of_occurrence" in df.columns and "date_reported" in df.columns:
    df["_year"] = df[["date_of_occurrence", "date_reported"]].min(axis=1).dt.year
elif "date_of_occurrence" in df.columns:
    df["_year"] = df["date_of_occurrence"].dt.year
elif "date_reported" in df.columns:
    df["_year"] = df["date_reported"].dt.year
df["_year"] = pd.to_numeric(df["_year"], errors="coerce").astype("Int64")

# Region detection
region_candidates = ["victim_district", "district", "state_name", "state"]
region_col = next((c for c in region_candidates if c in df.columns), None)
if region_col is None:
    st.error("No region column detected.")
    st.stop()

# Crime domain
domain_candidates = ["crime_domain", "crime_type", "crime_category", "crime_description", "category"]
domain_col = next((c for c in domain_candidates if c in df.columns), None)

# Crime count
if "crime_count" in df.columns:
    df["_total_crimes"] = df["crime_count"].fillna(0).astype(float)
else:
    count_candidates = [
        c for c in df.columns
        if any(k in c for k in ("report_number","report_id","case","incident","count"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    df["_total_crimes"] = df[count_candidates[0]].fillna(0).astype(float) if count_candidates else 1.0

# Load ML model (if exists)
MODEL = None
MODEL_PATHS = [
    Path("output/extratrees_ultrafast_v4.pkl"),
    Path("output/model_rf_classifier.joblib"),
    Path("output/model_rf_classifier.pkl"),
    Path("output/rf_classifier.joblib"),
    Path("output/extratrees_v4.pkl")
]
for p in MODEL_PATHS:
    if p.exists():
        try:
            if p.suffix in (".pkl", ".pickle"):
                import pickle
                with open(p, "rb") as f:
                    MODEL = pickle.load(f)
            else:
                MODEL = joblib.load(p)
            break
        except:
            MODEL = None

# Feature list
FEATURES = []
feat_path = Path("output/features_ultrafast_v4.json")
if not feat_path.exists():
    alt = Path("output/feature_config.json")
    if alt.exists(): feat_path = alt
if feat_path.exists():
    try: FEATURES = json.load(open(feat_path, "r")).get("features", [])
    except: FEATURES = []

# Visuals directory
VIS_DIR = Path("output/visuals")
VIS_DIR.mkdir(exist_ok=True)

# ---------------- Tabs ----------------
tabs = st.tabs([
    "Crime Domain Analysis",
    "Victim Demographics",
    "District Crime Activity",
    "Feature Importance & Relationships",
    "Correlation & Pairwise"
])

PLOTLY_HEIGHT = 320
MATPLOT_FIGSIZE = (6, 3.2)

# ---------------- TAB 1 ----------------
with tabs[0]:
    st.header("Crime Domain Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Crime Categories")
        if domain_col:
            counts = df[domain_col].fillna("Unknown").value_counts().reset_index()
            counts.columns = [domain_col, "count"]
            top_n = counts.head(10)
            fig = px.bar(top_n, x="count", y=domain_col, orientation="h",
                         color="count", color_continuous_scale="Reds", height=PLOTLY_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Domain column missing.")

    with col2:
        st.subheader("Composition (Pie)")
        if domain_col:
            fig = px.pie(counts.head(8), names=domain_col, values="count", height=PLOTLY_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pie chart unavailable.")


# ---------------- TAB 2 ----------------
with tabs[1]:
    st.header("Victim Demographics")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        if "victim_age" in df.columns:
            fig, ax = plt.subplots(figsize=MATPLOT_FIGSIZE)
            sns.histplot(df["victim_age"].dropna(), bins=30, kde=True, color="orange", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("victim_age missing.")

    with col2:
        st.subheader("Gender Breakdown")
        if "victim_gender" in df.columns:
            gender_counts = df["victim_gender"].fillna("Unknown").value_counts()
            fig = px.pie(values=gender_counts.values, names=gender_counts.index, height=PLOTLY_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("victim_gender missing.")


# ---------------- TAB 3 ----------------
with tabs[2]:
    st.header("District Crime Activity")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Districts")
        topd = df.groupby(region_col)["_total_crimes"].sum().sort_values(ascending=False).head(10).reset_index()
        topd.columns = [region_col, "count"]
        fig = px.bar(topd, x="count", y=region_col, orientation="h",
                     color="count", color_continuous_scale="Blues", height=PLOTLY_HEIGHT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("National Yearly Trend")
        if df["_year"].notna().any():
            yearly = df[df["_year"].notna()].groupby("_year")["_total_crimes"].sum().reset_index()
            fig = px.line(yearly, x="_year", y="_total_crimes", markers=True, height=PLOTLY_HEIGHT)
            fig.update_traces(line=dict(color="#00e0ff"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No year data.")


# ---------------- TAB 4 ----------------
with tabs[3]:
    st.header("Feature Importance & Relationships")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance")
        if MODEL is not None and hasattr(MODEL, "feature_importances_"):
            try:
                importances = MODEL.feature_importances_
                try:
                    names = list(MODEL.feature_names_in_)
                except:
                    names = FEATURES if FEATURES else [f"f_{i}" for i in range(len(importances))]
                fi = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
                fig = px.bar(fi.head(12), x="importance", y="feature",
                             orientation="h", height=PLOTLY_HEIGHT)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Model importance failed.")
        else:
            st.info("Model not found.")

    with col2:
        st.subheader("Boxplots by Crime Domain")
        if domain_col:
            top_feats = df.select_dtypes(include=[np.number]).columns[:4]
            for feat in top_feats:
                fig = px.box(df, x=domain_col, y=feat, height=200)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No domain column.")


# ---------------- TAB 5 ----------------
with tabs[4]:
    st.header("Correlation & Pairwise")
    col1, col2 = st.columns(2)

    numeric = df.select_dtypes(include=[np.number]).fillna(0)

    with col1:
        st.subheader("Correlation Heatmap")
        if numeric.shape[1] >= 2:
            corr = numeric.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", height=480)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns.")

    with col2:
        st.subheader("Top Correlated Pairs")
        if numeric.shape[1] >= 2:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = upper.stack().reset_index()
            pairs.columns = ["var1", "var2", "corr"]
            pairs["abs_corr"] = pairs["corr"].abs()
            top_pairs = pairs.sort_values("abs_corr", ascending=False).head(12)
            top_pairs["pair"] = top_pairs["var1"] + " ⟷ " + top_pairs["var2"]
            fig = px.bar(top_pairs, x="abs_corr", y="pair", orientation="h", height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data.")

    st.subheader("Scatter Matrix")
    try:
        var = numeric.var().sort_values(ascending=False)
        cols = var.head(6).index.tolist()
        sample = numeric[cols].dropna().sample(n=min(400, len(numeric)), random_state=42)
        if len(cols) >= 2:
            fig = px.scatter_matrix(sample, dimensions=cols, height=480)
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Scatter matrix too large or failed.")


# ---------------- Footer ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#bde0fe;'>Dashboard aligned with your FYP notebook.</p>", unsafe_allow_html=True)
