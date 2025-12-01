# utils/models/crime_model.py
# ------------------------------------------------------------
# This is the restructured version of your crime_model_full_pipeline.py
# Converted into a clean, importable module similar to full_model.py
# Compatible with Streamlit dashboard integration.
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Train Model Function
# ------------------------------------------------------------
def train_crime_model(df: pd.DataFrame, target="crime_domain", output_dir="models_output"):
    """
    Train Random Forest + Logistic Regression based on uploaded dataset.
    Saves both models and returns metrics.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset.")

    df[target] = df[target].astype(str).str.lower().str.strip()

    X = df.select_dtypes(include=["int64", "float64"]).fillna(0)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )
    rf.fit(X_train_res, y_train_res)
    y_pred_rf = rf.predict(X_test)

    # Train LR
    lr = LogisticRegression(max_iter=300, solver="lbfgs", class_weight="balanced")
    lr.fit(X_train_res, y_train_res)
    y_pred_lr = lr.predict(X_test)

    Path(output_dir).mkdir(exist_ok=True)
    rf_path = Path(output_dir) / "rf_model.joblib"
    lr_path = Path(output_dir) / "lr_model.joblib"
    joblib.dump(rf, rf_path)
    joblib.dump(lr, lr_path)

    # Metrics
    results = {
        "rf": {
            "accuracy": accuracy_score(y_test, y_pred_rf),
            "balanced_acc": balanced_accuracy_score(y_test, y_pred_rf),
            "f1": f1_score(y_test, y_pred_rf, average="weighted"),
            "report": classification_report(y_test, y_pred_rf, output_dict=True),
        },
        "lr": {
            "accuracy": accuracy_score(y_test, y_pred_lr),
            "balanced_acc": balanced_accuracy_score(y_test, y_pred_lr),
            "f1": f1_score(y_test, y_pred_lr, average="weighted"),
            "report": classification_report(y_test, y_pred_lr, output_dict=True),
        },
    }

    return results, rf_path, lr_path


# ------------------------------------------------------------
# 2. Predict Function
# ------------------------------------------------------------
def run_crime_prediction(df: pd.DataFrame, model_path: str):
    """
    Load saved model and run predictions on dataset.
    Returns a dataframe containing:
    - Actual crime domain
    - Predicted crime domain
    - Probabilities for each class
    """
    model = joblib.load(model_path)

    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    if "crime_domain" not in df.columns:
        raise KeyError("Dataset missing 'crime_domain' column.")

    X = df.select_dtypes(include=["int64", "float64"]).fillna(0)
    y = df["crime_domain"].astype(str).str.lower()

    preds = model.predict(X)
    probs = model.predict_proba(X)

    proba_df = pd.DataFrame(probs, columns=model.classes_)

    results = pd.DataFrame({
        "Actual_Crime_Domain": y,
        "Predicted_Crime_Domain": preds,
    })

    results = pd.concat([results.reset_index(drop=True), proba_df], axis=1)
    return results


# ------------------------------------------------------------
# 3. Feature Importance Extraction
# ------------------------------------------------------------
def get_feature_importance(model_path: str, feature_names):
    model = joblib.load(model_path)
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model has no feature_importances_ attribute.")

    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)
