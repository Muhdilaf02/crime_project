# ===============================================================
# 🌊 6_Explainability.py — CyberGlass Aqua Guided Explainability (ENHANCED)
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
import random
import time
from pathlib import Path

from utils.common import theme_css, status_badge
from utils.data_pipeline import load_cleaned_data
from utils.guidance import guide_box, steps_box

# UI extras
from utils.navbar import cyber_navbar
from utils.glow_panels import add_glow_effect

# Try import shap (optional)
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------
st.set_page_config(page_title="Model Explainability", page_icon="🧠", layout="wide")
st.markdown(theme_css(), unsafe_allow_html=True)
cyber_navbar("Explainability", user_name="Aila Farah", user_role="Admin")
add_glow_effect()

# Styles (kept from your original)
st.markdown("""
<style>
h1 { text-align:center; font-weight:900; color:#00e0ff; font-size:46px; text-shadow:0 0 25px #00b4d8; }
.cy-divider { height:2px; background:linear-gradient(90deg,transparent,#00b4d8,transparent); margin:25px 0; border-radius:2px; }
.card { background: rgba(255,255,255,0.07); border: 1px solid rgba(0,255,255,0.15); border-radius: 16px; padding: 18px 22px; backdrop-filter: blur(12px); box-shadow: 0 0 25px rgba(0,180,216,0.25); transition: all 0.3s ease; }
.card:hover { transform: scale(1.01); box-shadow: 0 0 35px rgba(0,200,255,0.35); }
.metric { font-size:34px; font-weight:800; color:#00e0ff; text-shadow:0 0 12px #00b4d8; }
.story { background:rgba(0,0,0,0.35); border-left:4px solid #00e0ff; padding:14px 18px; border-radius:10px; margin-top:12px; color:#caf0f8; }
.story b { color:#00e0ff; }
.explain { color:#8ecae6; font-size:15px; font-style:italic; margin-top:5px; }
.small { font-size:13px; color:#9fd3e6; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1>🧠 Model Explainability & What-If Analysis</h1>
<p style="text-align:center;color:#ade8f4;font-size:18px;">
Understand how your prediction model makes decisions — step by step and with interactive what-if simulations.
</p>
<div class="cy-divider"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# INTRO & GUIDANCE
# ---------------------------------------------------------------
guide_box(
    "🔍 What This Page Does",
    """
This section helps users **understand how the prediction model thinks**:

- See which features influence predictions the most (global importance).  
- Run interactive "What-If" scenarios (adjust features and observe changes).  
- Get SHAP explanations for instance-level reasoning (if available).  
- Save scenario inputs & outputs for documentation.
""",
    icon="💡",
    open_default=True
)

steps_box(
    "🪜 Step-by-Step Guide",
    [
        "Load and verify the trained models and dataset.",
        "View top global feature importances.",
        "Adjust variables to simulate different scenarios (What-If).",
        "See immediate model predictions and local explanations.",
        "Export results for inclusion in your FYP report."
    ],
    icon="🧭",
    open_default=False
)

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
try:
    df = load_cleaned_data()
    # normalize
    df.columns = df.columns.str.lower().str.strip()
    status_badge("✅ Dataset Loaded", "ok")
except Exception as e:
    status_badge("❌ Failed to load dataset", "err")
    st.error(f"Failed to load cleaned dataset: {e}")
    st.stop()

# ---------------------------------------------------------------
# LOAD MODELS & FEATURES
# ---------------------------------------------------------------
model_dir = Path("C:/Users/ilafl/projects/crime_project/output")
regressor = None
classifier = None
feature_cfg = {}

# Attempt to load expected files with robust error messages
try:
    with open(model_dir / "regressor.pkl", "rb") as f:
        regressor = pickle.load(f)
    with open(model_dir / "classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open(model_dir / "feature_config.json", "r", encoding="utf-8") as f:
        feature_cfg = json.load(f)
    features = feature_cfg.get("features", [])
    status_badge("✅ Models Loaded", "ok")
except Exception as e:
    status_badge("⚠️ Models not found", "warn")
    st.error("Please ensure 'regressor.pkl', 'classifier.pkl', and 'feature_config.json' exist in the output folder.")
    st.stop()

# Ensure features is a list
if not isinstance(features, list) or len(features) == 0:
    st.warning("Feature list seems empty — model inputs will be inferred where possible.")
    # fallback: numeric columns from df
    features = [c for c in df.select_dtypes(include=[np.number]).columns.tolist()]

# ---------------------------------------------------------------
# CHECK FEATURE PRESENCE
# ---------------------------------------------------------------
missing_feats = [f for f in features if f not in df.columns]
if missing_feats:
    st.warning(f"⚠️ {len(missing_feats)} model features not found in dataset (they will be filled with 0).")
else:
    st.caption("✅ All model features present in the dataset.")

# ---------------------------------------------------------------
# STEP 1: GLOBAL FEATURE IMPORTANCE (Plot + explanation)
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("Step 1 — Global Feature Importance")
st.caption("Which features the regressor considers most important when predicting crime intensity (global view).")

# compute importance with fallback
imp_ser = None
if hasattr(regressor, "feature_importances_"):
    try:
        imp_ser = pd.Series(regressor.feature_importances_, index=features).sort_values(ascending=False)
    except Exception:
        # attempt to map feature_importances_ to model.feature_names_in_
        try:
            names = list(regressor.feature_names_in_)
            imp_ser = pd.Series(regressor.feature_importances_, index=names).sort_values(ascending=False)
        except Exception:
            imp_ser = None

if imp_ser is None:
    # fallback: use variance across candidate features as proxy
    numeric_candidates = [c for c in features if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    if numeric_candidates:
        imp_ser = df[numeric_candidates].var().sort_values(ascending=False)
        imp_ser.name = "variance_proxy"
        imp_ser = imp_ser
        st.info("Model importance not available — using variance as a proxy for global influence.")
    else:
        st.error("Cannot compute any feature importance (no numeric features).")

if imp_ser is not None:
    top_imp = imp_ser.head(12)
    fig_imp = px.bar(
        x=top_imp.values[::-1],
        y=top_imp.index[::-1],
        orientation="h",
        labels={"x": "Importance", "y": "Feature"},
        title="Top Features (Global Importance)",
        height=360,
        color=top_imp.values[::-1],
        color_continuous_scale="tealgrn"
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("<p class='explain'>Higher values indicate greater global influence on the model's predictions (or greater variability in the proxy case).</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# STEP 2: WHAT-IF Sliders (Interactive inputs)
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("Step 2 — Interactive What-If Scenario")
st.caption("Adjust feature values to see how the model prediction and explanations change in real time.")

# Allow user to choose how many sliders to show (1..8)
max_sliders = st.slider("Number of features to simulate (sliders)", min_value=1, max_value=8, value=5)

# Choose candidate numeric features (intersection of features & dataframe numeric columns)
numeric_features = [c for c in features if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
# If too few, expand to top variance numeric columns from df
if len(numeric_features) < max_sliders:
    extra = [c for c in df.select_dtypes(include=[np.number]).columns if c not in numeric_features]
    numeric_features = numeric_features + extra
numeric_features = list(dict.fromkeys(numeric_features))  # dedupe preserving order

# choose default sliders: top by importance (if available) else by variance
if imp_ser is not None:
    candidate_order = [f for f in imp_ser.index if f in numeric_features]
else:
    # fallback by variance
    candidate_order = list(pd.Series(df[numeric_features].var()).sort_values(ascending=False).index)

slider_features = candidate_order[:max_sliders]

# layout sliders in two columns
col_a, col_b = st.columns(2)
user_vals = {}
for i, feat in enumerate(slider_features):
    col = col_a if i % 2 == 0 else col_b
    col.markdown(f"**{feat.replace('_',' ').title()}**")
    # compute min, max, median safely
    col_min = float(np.nanmin(df[feat].dropna())) if feat in df.columns else 0.0
    col_max = float(np.nanmax(df[feat].dropna())) if feat in df.columns else (col_min + 1)
    col_mid = float(np.nanmedian(df[feat].dropna())) if feat in df.columns else col_min
    step = (col_max - col_min) / 100 if (col_max - col_min) != 0 else 1.0
    # Use number_input for precise values when ranges are small
    user_vals[feat] = col.slider("", min_value=col_min, max_value=col_max, value=col_mid, step=step)

# Prepare aligned input for model: fill missing features with 0
if user_vals:
    X_inst = pd.DataFrame([ {f: user_vals.get(f, 0.0) for f in features} ])
    X_inst = X_inst.reindex(columns=features, fill_value=0.0)
else:
    X_inst = pd.DataFrame(columns=features)

# Option to save the scenario
save_scenario = st.checkbox("Enable saving this scenario to JSON", value=False)
if save_scenario:
    scenario_name = st.text_input("Filename (without extension)", value=f"scenario_{int(time.time())}")
    save_button = st.button("Save scenario now")

# ---------------------------------------------------------------
# STEP 3: RUN PREDICTION & LOCAL EXPLANATIONS
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("Step 3 — Prediction & Local Explanation")
st.caption("Run the model on the scenario and view SHAP explanations (if available) or feature contribution fallback.")

col1, col2, col3 = st.columns([2,2,2])

run_btn = col3.button("Run Prediction")

# function to compute SHAP (if available) or fallback local contributions
def compute_local_explanation(model, X_row):
    explanation = {}
    # Try SHAP TreeExplainer for tree models
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_row)
            # shap_values shape differs by classifier/regressor: for regressor single array
            explanation["shap_values"] = shap_values
            explanation["explainer"] = "shap"
            return explanation
        except Exception:
            # fallback to kernel explainer for other models (may be slow) — we skip kernel automatically
            return {"shap_values": None, "explainer": None}
    else:
        return {"shap_values": None, "explainer": None}

# run prediction block
if run_btn:
    try:
        # regressor prediction
        pred_val = float(regressor.predict(X_inst)[0])
        # classifier prediction & prob if available
        try:
            cls_pred = classifier.predict(X_inst)[0]
            cls_proba = classifier.predict_proba(X_inst)[0]
            highest_class = classifier.classes_[np.argmax(cls_proba)]
            confidence = float(np.max(cls_proba) * 100)
        except Exception:
            cls_pred = "N/A"
            highest_class = "N/A"
            confidence = 0.0

        # display metrics
        col1.metric("Predicted Intensity", f"{pred_val:.2f}")
        col2.metric("Predicted Class", str(cls_pred))
        col3.metric("Confidence", f"{confidence:.1f}%")

        # Save scenario if requested
        if save_scenario and 'save_button' in locals() and save_button:
            out = {
                "timestamp": int(time.time()),
                "scenario_name": scenario_name,
                "input": X_inst.iloc[0].to_dict(),
                "prediction": {"regression": pred_val, "classification": str(cls_pred), "confidence_pct": confidence}
            }
            outfile = (model_dir / f"{scenario_name}.json")
            with open(outfile, "w", encoding="utf-8") as fo:
                json.dump(out, fo, indent=2)
            st.success(f"Saved scenario to: {outfile}")

        # compute local explanation
        explanation = compute_local_explanation(regressor, X_inst)
        if explanation.get("explainer") == "shap" and explanation.get("shap_values") is not None:
            try:
                # For regressor shap_values is array
                shap_vals = explanation["shap_values"]
                # Convert to DataFrame of absolute contributions
                if isinstance(shap_vals, list):
                    # Multi-output: take first
                    shap_vals_arr = shap_vals[0]
                else:
                    shap_vals_arr = shap_vals
                # shap_vals_arr shape (n_samples, n_features)
                sv = pd.Series(np.abs(shap_vals_arr[0]), index=features).sort_values(ascending=False).head(10)
                st.subheader("Top local SHAP contributions (absolute)")
                fig_shap = px.bar(x=sv.values[::-1], y=sv.index[::-1], orientation="h", height=360, color=sv.values[::-1], color_continuous_scale="tealgrn")
                st.plotly_chart(fig_shap, use_container_width=True)
                st.markdown("<p class='explain'>SHAP shows how each feature pushed the model prediction up or down for this specific instance.</p>", unsafe_allow_html=True)
            except Exception as e:
                st.info("SHAP computation produced unexpected result: " + str(e))
        else:
            # Fallback local contribution: simple difference from mean * importance proxy
            st.subheader("Local contribution (fallback)")
            # Build proxy scores: (value - mean) * global importance (normalized)
            val = X_inst.iloc[0]
            global_imp = imp_ser if 'imp_ser' in locals() and imp_ser is not None else pd.Series(df[numeric_features].var(), index=numeric_features)
            # align
            aligned_imp = global_imp.reindex(index=features).fillna(0.0)
            deltas = (val - df[features].mean()).abs() * aligned_imp
            deltas = deltas.sort_values(ascending=False).head(10)
            fig_loc = px.bar(x=deltas.values[::-1], y=deltas.index[::-1], orientation="h", height=360, color=deltas.values[::-1], color_continuous_scale="tealgrn")
            st.plotly_chart(fig_loc, use_container_width=True)
            st.markdown("<p class='explain'>This approximate local contribution shows which changed features (relative to dataset mean) combined with global importance produce the largest influence.</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction / explanation failed: {e}")

# ---------------------------------------------------------------
# STEP 4: Human-friendly explanation (auto-generated)
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("Step 4 — Human-friendly Explanation")

if user_vals:
    # pick top 2 contributing features from last computed deltas or shap
    try:
        top_two = deltas.head(2).index.tolist() if 'deltas' in locals() else (list(user_vals.keys())[:2])
    except Exception:
        top_two = list(user_vals.keys())[:2]
    if len(top_two) == 0:
        st.info("No input features available to generate a short explanation.")
    else:
        # deterministic "story" text (not random)
        a, b = top_two[0], (top_two[1] if len(top_two) > 1 else None)
        story_lines = []
        story_lines.append(f"In this scenario, the model shows notable sensitivity to **{a.replace('_',' ')}**.")
        if b:
            story_lines.append(f"**{b.replace('_',' ')}** also contributes, but to a lesser extent.")
        story_lines.append("Together these variables influence the predicted crime intensity for the selected district/year profile.")
        story_text = "<br>".join(story_lines)
        st.markdown(f"<div class='story'>{story_text}</div>", unsafe_allow_html=True)
        st.markdown("<p class='explain'>This summary is intended for non-technical readers and can be copy-pasted into your FYP report.</p>", unsafe_allow_html=True)
else:
    st.info("No scenario inputs to summarise. Use the sliders above and run a prediction.")

# ---------------------------------------------------------------
# DOWNLOAD / SAVE RESULTS utility
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("Export / Save")
st.caption("Download the last scenario (input + prediction + explanation) as JSON for documentation.")

if 'out' in locals():
    json_bytes = json.dumps(out, indent=2).encode("utf-8")
    st.download_button("Download Last Scenario JSON", data=json_bytes, file_name="scenario_output.json", mime="application/json")
else:
    st.info("Run a prediction first to enable download.")

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("""
<div class="cy-divider"></div>
<div style="text-align:center;color:#8ecae6;font-size:14px;">
© 2025 Crime Analytics Dashboard — Explainability Module • CyberGlass Aqua
</div>
""", unsafe_allow_html=True)
