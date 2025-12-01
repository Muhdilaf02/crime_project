# app_prediction_pro.py
# ===============================================================
# CyberGlass — Crime Risk Prediction (Professional Streamlit UI)
# - Loads ExtraTrees UltraFast model
# - District × Year auto-fill
# - Simple Low/Med/High dropdowns for input
# - Probability bar + radar chart
# - SHAP explanation (fallback to feature_importances_)
# - Save JSON export
# ===============================================================

import json
import time
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional: shap may not be installed — we try/except
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------------- CONFIG ----------------------
DATA_PATH = Path(r"C:\Users\ilafl\projects\crime_project\output\final_cleaned_crime_socioeconomic_data.csv")
MODEL_PATH = Path(r"C:\Users\ilafl\projects\crime_project\output\extratrees_ultrafast_v4.pkl")
FEATURES_PATH = Path(r"C:\Users\ilafl\projects\crime_project\output\features_ultrafast_v4.json")
OUTPUT_DIR = Path(r"C:\Users\ilafl\projects\crime_project\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="CyberGlass — Crime Risk (Pro)", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #061126 0%, #0b2236 100%); color: #e6f7ff; }
    .card { background: rgba(255,255,255,0.04); padding:20px; border-radius:12px; }
    .title { font-family: 'Inter', sans-serif; color:#00e0ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='title' style='text-align:center'>🛡️ CyberGlass — Crime Risk Prediction (Professional)</h1>", unsafe_allow_html=True)
st.write("Fast • Simple • Explainable — District × Year panel model (ExtraTrees UltraFast)")

# ---------------------- UTILITY ----------------------
def load_model(path):
    if not path.exists():
        st.error(f"Model not found: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def load_features(path):
    if not path.exists():
        st.error(f"Feature config not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f).get("features", [])

def aggregate_district_year(df):
    # minimal aggregation similar to training pipeline
    agg = (
        df.groupby(["state_name","district","year"], as_index=False)
        .agg({c: "median" for c in df.select_dtypes(include=[np.number]).columns if c not in ["year","state_code","district_code"]})
    )
    # ensure crime_count exists: try count of report_number if present
    if "report_number" in df.columns:
        count = df.groupby(["state_name","district","year"], as_index=False)["report_number"].count().rename(columns={"report_number":"crime_count"})
        agg = agg.merge(count, on=["state_name","district","year"], how="left")
    else:
        agg["crime_count"] = agg.get("crime_count", 0)
    return agg

def align_features(input_row: pd.DataFrame, features_list):
    # Ensure all features exist and in correct order
    df = input_row.copy()
    for f in features_list:
        if f not in df.columns:
            df[f] = 0.0
    return df[features_list].astype(float)

def save_prediction_json(out_path: Path, payload: dict):
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

# ---------------------- LOAD MODEL & DATA ----------------------
with st.spinner("Loading model and data..."):
    model = load_model(MODEL_PATH)
    FEATURES = load_features(FEATURES_PATH)
    try:
        raw_df = pd.read_csv(DATA_PATH, low_memory=False)
        # clean column names
        raw_df.columns = raw_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w\d_]", "", regex=True)
        if "date_of_occurrence" in raw_df.columns:
            raw_df["date_of_occurrence"] = pd.to_datetime(raw_df["date_of_occurrence"], errors="coerce", dayfirst=True)
            raw_df["year"] = raw_df["date_of_occurrence"].dt.year
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        raw_df = pd.DataFrame()

if model is None or not FEATURES or raw_df.empty:
    st.stop()

st.success("Model & data loaded")

# ---------------------- SIDEBAR INPUT ----------------------
st.sidebar.markdown("### 🔧 Settings")
auto_fill = st.sidebar.checkbox("Auto-fill from historical district-year data", value=True)
show_shap = st.sidebar.checkbox("Show SHAP explainability (if available)", value=True)
save_predictions = st.sidebar.checkbox("Enable saving predictions (JSON)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick tips**: Use Auto-fill to pre-populate socio-economic features for the chosen district & year. If SHAP fails, feature_importances will be shown instead.")

# ---------------------- REGION & YEAR SELECT ----------------------
st.subheader("📍 Choose Region & Year")
col1, col2, col3 = st.columns([3,3,2])
with col1:
    region_type = st.radio("Region Type", ["District", "State"], horizontal=True)
with col2:
    if region_type == "District":
        regions = sorted(raw_df["district"].dropna().unique())
    else:
        regions = sorted(raw_df["state_name"].dropna().unique())
    region = st.selectbox("Select region", options=regions, index=0)
with col3:
    years = sorted(raw_df["year"].dropna().unique())
    if len(years) == 0:
        st.error("No year info in dataset.")
        st.stop()
    year = st.selectbox("Year", options=years, index=len(years)-1)

# ---------------------- INPUT PROFILES (Low/Med/High) ----------------------
st.subheader("📊 Choose Profile (Simple dropdowns)")
profile_col1, profile_col2, profile_col3 = st.columns(3)

mapping = {"Low": 30, "Medium": 60, "High": 90}
def map_val(s): return mapping.get(s, 50)

with profile_col1:
    poverty = st.selectbox("Poverty level", ["Low","Medium","High"], index=1)
    literacy = st.selectbox("Literacy level", ["Low","Medium","High"], index=1)
with profile_col2:
    unemployment = st.selectbox("Unemployment level", ["Low","Medium","High"], index=1)
    toilet_access = st.selectbox("Toilet access (proxy)", ["Low","Medium","High"], index=1)
with profile_col3:
    vehicle = st.selectbox("Motor vehicle ownership", ["Low","Medium","High"], index=1)
    govt_emp = st.selectbox("Government employee households", ["Low","Medium","High"], index=1)

# ---------------------- BUILD INPUT ROW ----------------------
# First, build a minimal input based on profile choices (mapping names to plausible feature names)
# NOTE: These keys must match some features in FEATURES (if not, they will be filled with zeros)
base_input = {
    "poverty_rate_(%)": map_val(poverty),
    "literacy_rate_(%)": map_val(literacy),
    "unemployment_rate_(%)": map_val(unemployment),
    "own_motor_vehicle": map_val(vehicle),
    "own_refrigerator": map_val(toilet_access),   # reuse for a household asset proxy
    "govt_employee_member": map_val(govt_emp)
}

# If auto_fill, try to pull aggregated district-year row and fill additional features
agg_df = aggregate_district_year(raw_df) if not raw_df.empty else pd.DataFrame()
prefill_row = {}
if auto_fill and not agg_df.empty:
    # for district-level selection, find exact match (state+district-year) or fallback to district median
    if region_type == "District":
        subset = agg_df[(agg_df["district"] == region) & (agg_df["year"] == int(year))]
        if subset.empty:
            # fallback: use most recent available for that district
            subset = agg_df[agg_df["district"] == region].sort_values("year", ascending=False).head(1)
        if subset.empty:
            # fallback: median across all districts
            subset = agg_df.head(1)
    else:
        subset = agg_df[(agg_df["state_name"] == region) & (agg_df["year"] == int(year))]
        if subset.empty:
            subset = agg_df[agg_df["state_name"] == region].sort_values("year", ascending=False).head(1)
        if subset.empty:
            subset = agg_df.head(1)

    if not subset.empty:
        row = subset.iloc[0].to_dict()
        # choose numeric columns from this row to prefill
        for k,v in row.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and k not in ["year"]:
                prefill_row[k] = v

# Merge profile base_input and prefill; prefill keys take precedence
merged_input = {**base_input, **prefill_row}

# Create DataFrame for input
X_input = pd.DataFrame([merged_input])
st.markdown("### 🔎 Input preview (features used by model)")
st.dataframe(X_input[list(X_input.columns)].T.rename(columns={0:"value"}), height=300)

# ---------------------- ALIGN FEATURES & PREDICT ----------------------
# Align features to model's expectation
X_aligned = align_features(X_input, FEATURES)

st.markdown("---")
colL, colR = st.columns([2,1])
with colL:
    st.write("Ready to predict using aligned features (missing filled by 0).")
with colR:
    run = st.button("🚀 Run Prediction", use_container_width=True)

if run:
    t0 = time.time()
    try:
        pred_label = model.predict(X_aligned)[0]
        proba = model.predict_proba(X_aligned)[0]
        classes = model.classes_
        proba_series = pd.Series(proba, index=classes).sort_values(ascending=False)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    duration = time.time() - t0

    # ---------------------- RESULTS PANEL ----------------------
    st.markdown("## ✅ Prediction Result")
    rcol1, rcol2 = st.columns([2,3])

    with rcol1:
        st.metric("Predicted Risk Level", f"{pred_label}")
        st.metric("Inference time (s)", f"{duration:.3f}")

        # probability bar
        p_df = pd.DataFrame({"Risk": proba_series.index, "Probability": proba_series.values})
        fig_bar = px.bar(p_df, x="Probability", y="Risk", orientation="h", color="Probability",
                         color_continuous_scale="RdYlGn_r", labels={"Probability":"Probability"})
        st.plotly_chart(fig_bar, use_container_width=True)

        # save JSON
        if save_predictions:
            save_name = OUTPUT_DIR / f"prediction_{region}_{int(year)}_{int(time.time())}.json"
            payload = {
                "timestamp": int(time.time()),
                "region": region,
                "region_type": region_type,
                "year": int(year),
                "input": merged_input,
                "predicted_label": str(pred_label),
                "probabilities": proba_series.to_dict()
            }
            save_prediction_json(save_name, payload)
            st.success(f"Saved prediction: {save_name.name}")

    with rcol2:
        # radar/polar chart for probabilities
        theta = list(proba_series.index)
        r = list(proba_series.values)
        fig = go.Figure(data=go.Scatterpolar(r=r + [r[0]], theta=theta + [theta[0]], fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False, title="Probability Radar")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------- SHAP EXPLAINABILITY ----------------------
    st.markdown("### 🔬 Explainability")
    if SHAP_AVAILABLE and show_shap:
        try:
            # For tree models use TreeExplainer
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_aligned)
            # shap_values: list per class
            # find class index
            class_index = list(model.classes_).index(pred_label)
            shap_vals_for_class = sv[class_index]
            # build df of top features for the instance
            instance_shap = pd.Series(shap_vals_for_class[0], index=X_aligned.columns)
            top_feats = instance_shap.abs().sort_values(ascending=False).head(10)
            shap_df = pd.DataFrame({
                "feature": top_feats.index,
                "shap_value": instance_shap[top_feats.index].values
            }).sort_values("shap_value", key=abs, ascending=False)
            # plot bar
            fig2 = px.bar(shap_df, x="shap_value", y="feature", orientation="h", title="Top SHAP features (instance)")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP explainability failed: {e}. Falling back to feature_importances_.")
            # fallback to model.feature_importances_
            fi = pd.Series(model.feature_importances_, index=X_aligned.columns).sort_values(ascending=False).head(10)
            fig3 = px.bar(x=fi.values, y=fi.index, orientation="h", title="Top feature importances (model)")
            st.plotly_chart(fig3, use_container_width=True)
    else:
        # fallback: show model feature_importances_
        fi = pd.Series(model.feature_importances_, index=X_aligned.columns).sort_values(ascending=False).head(10)
        fig3 = px.bar(x=fi.values, y=fi.index, orientation="h", title="Top feature importances (model)")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.info("Tip: Use Auto-fill for a realistic district-year profile; tweak dropdowns to test scenarios.")

# End of app
