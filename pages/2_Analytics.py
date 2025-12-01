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
import json, joblib, warnings
warnings.filterwarnings("ignore")

# UI helpers from your project
from utils.common import theme_css, get_paths
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

# ---------------- Load dataset ----------------
paths = get_paths()
DATA_DIR = paths["OUTPUT"]
CSV = DATA_DIR / "final_cleaned_crime_socioeconomic_data.csv"

if not CSV.exists():
    st.error(f"Dataset not found at: {CSV}")
    st.stop()

# read
df = pd.read_csv(CSV, low_memory=False)
# normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w\d_]", "", regex=True)

# parse date columns and build _year like notebook
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

# region col detection
region_candidates = ["victim_district", "district", "state_name", "state"]
region_col = next((c for c in region_candidates if c in df.columns), None)
if region_col is None:
    st.error("No region column detected (looked for victim_district, district, state_name, state).")
    st.stop()

# crime domain detection (multiple fallbacks)
domain_candidates = ["crime_domain", "crime_type", "crime_category", "crime_description", "category"]
domain_col = next((c for c in domain_candidates if c in df.columns), None)

# total crimes logic
if "crime_count" in df.columns:
    df["_total_crimes"] = df["crime_count"].fillna(0).astype(float)
else:
    # attempt count-like columns
    count_candidates = [c for c in df.columns if any(k in c for k in ("report_number","report_id","case","incident","count")) and pd.api.types.is_numeric_dtype(df[c])]
    if count_candidates:
        df["_total_crimes"] = df[count_candidates[0]].fillna(0).astype(float)
    else:
        df["_total_crimes"] = 1.0  # fallback

# try load model for feature importance
MODEL = None
MODEL_PATHS = [
    DATA_DIR / "extratrees_ultrafast_v4.pkl",
    DATA_DIR / "model_rf_classifier.joblib",
    DATA_DIR / "model_rf_classifier.pkl",
    DATA_DIR / "rf_classifier.joblib",
    DATA_DIR / "extratrees_v4.pkl"
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
        except Exception:
            MODEL = None

# features list if present
FEATURES = []
feat_path = DATA_DIR / "features_ultrafast_v4.json"
if not feat_path.exists():
    alt = DATA_DIR / "feature_config.json"
    if alt.exists():
        feat_path = alt
if feat_path.exists():
    try:
        FEATURES = json.load(open(feat_path, "r")).get("features", [])
    except Exception:
        FEATURES = []

# create visuals dir
VIS_DIR = DATA_DIR / "visuals"
VIS_DIR.mkdir(exist_ok=True)

# ---------------- Tabs: 5 tabs, each with 2 columns ----------------
tabs = st.tabs([
    "Crime Domain Analysis",
    "Victim Demographics",
    "District Crime Activity",
    "Feature Importance & Relationships",
    "Correlation & Pairwise"
])

# helper for compact plot height
PLOTLY_HEIGHT = 320
MATPLOT_FIGSIZE = (6, 3.2)  # width, height in inches

# ---------------- TAB 1: Crime Domain Analysis ----------------
with tabs[0]:
    st.header("Crime Domain Analysis")
    col1, col2 = st.columns(2)
    # left: top bar
    with col1:
        st.subheader("Top Crime Categories (Bar)")
        if domain_col:
            counts = df[domain_col].fillna("Unknown").value_counts().reset_index()
            counts.columns = [domain_col, "count"]
            top_n = counts.head(10)
            fig = px.bar(top_n, x="count", y=domain_col, orientation="h",
                         color="count", color_continuous_scale="Reds", height=PLOTLY_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No crime-domain column found in dataset.")

        st.markdown("""
        **Explanation:** This horizontal bar chart displays the most frequently recorded crime categories.
        It helps prioritise intervention and shows which classes the predictive model needs to distinguish best.
        """)

    # right: pie for composition
    with col2:
        st.subheader("Composition (Pie)")
        if domain_col:
            topn = counts.head(8)
            fig2 = px.pie(topn, names=domain_col, values="count", height=PLOTLY_HEIGHT)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Pie chart unavailable (no domain column).")

        st.markdown("""
        **Explanation:** The pie chart complements the bar chart by visualising proportional share
        of top crime categories, useful for quick stakeholder communication.
        """)

# ---------------- TAB 2: Victim Demographics ----------------
with tabs[1]:
    st.header("Victim Demographics")
    col1, col2 = st.columns(2)

    # left: age distribution
    with col1:
        st.subheader("Age Distribution (Histogram)")
        if "victim_age" in df.columns:
            fig, ax = plt.subplots(figsize=MATPLOT_FIGSIZE)
            sns.histplot(df["victim_age"].dropna(), bins=30, kde=True, color="orange", ax=ax)
            ax.set_title("Victim Age Distribution")
            st.pyplot(fig)
        else:
            st.warning("victim_age column not found.")

        st.markdown("""
        **Explanation:** The histogram indicates which age groups are most affected by crime.
        These demographic patterns are important features for predictive modeling.
        """)

    # right: gender pie
    with col2:
        st.subheader("Gender Breakdown")
        if "victim_gender" in df.columns:
            gender_counts = df["victim_gender"].fillna("Unknown").value_counts()
            fig = px.pie(values=gender_counts.values, names=gender_counts.index, height=PLOTLY_HEIGHT,
                         title="Victim Gender")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("victim_gender column missing.")

        st.markdown("""
        **Explanation:** Gender distribution can reveal vulnerability patterns and influence feature selection.
        """)

# ---------------- TAB 3: District Crime Activity ----------------
with tabs[2]:
    st.header("District Crime Activity")
    col1, col2 = st.columns(2)

    # left: top districts bar
    with col1:
        st.subheader("Top Districts (Bar)")
        topd = df.groupby(region_col)["_total_crimes"].sum().sort_values(ascending=False).head(10).reset_index()
        topd.columns = [region_col, "count"]
        fig = px.bar(topd, x="count", y=region_col, orientation="h", color="count",
                     color_continuous_scale="Blues", height=PLOTLY_HEIGHT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Explanation:** Identifies hotspots by aggregate counts. Use this to target
        districts for detailed investigation and resource prioritisation.
        """)

    # right: national yearly trend (compact)
    with col2:
        st.subheader("National Yearly Trend")
        if "_year" in df.columns and df["_year"].notna().any():
            yearly = df[df["_year"].notna()].groupby("_year")["_total_crimes"].sum().reset_index()
            fig = px.line(yearly, x="_year", y="_total_crimes", markers=True, height=PLOTLY_HEIGHT,
                          title="National Yearly Crime Trend")
            fig.update_traces(line=dict(color="#00e0ff"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Explanation:** Shows trends over time (increase/decrease); useful to validate temporal patterns.
            """)
        else:
            st.info("No _year data to plot national trend.")

# ---------------- TAB 4: Feature Importance & Relationships ----------------
with tabs[3]:
    st.header("Feature Importance & Relationships")
    col1, col2 = st.columns(2)

    # left: feature importance (model or fallback variance)
    with col1:
        st.subheader("Feature Importance (Top 12)")
        if MODEL is not None and hasattr(MODEL, "feature_importances_"):
            try:
                importances = MODEL.feature_importances_
                try:
                    feature_names = list(MODEL.feature_names_in_)
                except Exception:
                    feature_names = FEATURES if FEATURES else [f"f_{i}" for i in range(len(importances))]
                fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
                topfi = fi.head(12)
                fig = px.bar(topfi, x="importance", y="feature", orientation="h", height=PLOTLY_HEIGHT,
                             title="Model Feature Importances")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Failed to render model feature importance: " + str(e))
        else:
            # fallback: show variance of selected socioeconomic features as proxy
            socio = [c for c in ["no_literate_adult_25_plus","mon_inc_lt_5k","govt_employee_member","sc_st_hh","landless_hh_manual_labor"] if c in df.columns]
            if socio:
                var = df[socio].var().sort_values(ascending=False).reset_index()
                var.columns = ["feature", "importance"]
                fig = px.bar(var.head(12), x="importance", y="feature", orientation="h", height=PLOTLY_HEIGHT,
                             title="Feature variability (proxy importance)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No model or socioeconomic columns available for importance.")

        st.markdown("""
        **Explanation:** Ranks features by importance (or variability). Important features inform which
        socioeconomic indicators to prioritise in predictive modeling.
        """)

    # right: boxplots of top features vs crime domain
    with col2:
        st.subheader("Boxplots: Top Features vs Crime Domain")
        domain = domain_col
        if domain is None:
            st.info("No crime-domain column; boxplots unavailable.")
        else:
            # determine top features to plot
            top_feats = []
            if MODEL is not None and hasattr(MODEL, "feature_importances_"):
                try:
                    feature_names = list(MODEL.feature_names_in_)
                    top_feats = [f for f in feature_names if f in df.columns][:4]
                except Exception:
                    top_feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("_year","data","id")][:4]
            else:
                top_feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("_year","data","id")][:4]

            if not top_feats:
                st.info("No numeric features available for boxplots.")
            else:
                # create small subplots
                for feat in top_feats:
                    fig = px.box(df, x=domain, y=feat, points="outliers", height=200, title=f"{feat} vs {domain}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"**Explanation:** Compares **{feat}** across crime types to show distribution differences.")

# ---------------- TAB 5: Correlation & Pairwise ----------------
with tabs[4]:
    st.header("Correlation & Pairwise Analysis")
    col1, col2 = st.columns(2)

    # left: correlation heatmap
    with col1:
        st.subheader("Correlation Heatmap")
        numeric = df.select_dtypes(include=[np.number]).fillna(0)
        if numeric.shape[1] < 2:
            st.info("Not enough numeric columns for correlation.")
        else:
            corr = numeric.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", height=480)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Explanation:** Shows linear relationships between numeric variables. Useful for spotting
            multicollinearity and strong predictive signals.
            """)

    # right: top correlated pairs as bar chart (compact)
    with col2:
        st.subheader("Top Correlated Pairs (Bar)")
        if numeric.shape[1] < 2:
            st.info("Not enough numeric columns.")
        else:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = (upper.stack().reset_index().rename(columns={0:"corr","level_0":"var1","level_1":"var2"}))
            pairs["abs_corr"] = pairs["corr"].abs()
            top_pairs = pairs.sort_values("abs_corr", ascending=False).head(12)
            # convert to bar by combining var1-var2 label
            top_pairs["pair"] = top_pairs["var1"] + " ⟷ " + top_pairs["var2"]
            fig = px.bar(top_pairs, x="abs_corr", y="pair", orientation="h", height=PLOTLY_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Explanation:** Top correlated variable pairs indicate strong linear relationships; inspect these
            to remove redundancy or to engineer interaction features.
            """)

        # pairplot alternative (scatter matrix) below pair bar (small)
        st.subheader("Scatter matrix (sample)")
        try:
            # choose up to 6 highest variance numeric columns
            var = numeric.var().sort_values(ascending=False)
            cols = var.head(6).index.tolist()
            sample = numeric[cols].dropna().sample(n=min(400, len(numeric)), random_state=42)
            if len(cols) >= 2:
                fig = px.scatter_matrix(sample, dimensions=cols, height=480)
                fig.update_traces(diagonal_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("Scatter matrix generation failed or too heavy: " + str(e))

# ---------------- Footer ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#bde0fe;'>Dashboard aligned with the FYP analysis notebook: exploratory visuals, hotspot detection and socioeconomic relationships to support predictive modelling.</p>", unsafe_allow_html=True)
