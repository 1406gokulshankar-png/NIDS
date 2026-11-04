# explain.py
# Purpose: use SHAP to find which features pushed one alert toward "attack".

import joblib
import numpy as np
import shap

from preprocess import preprocess, CATEGORICAL_COLUMNS

# We load the data and model only once, then keep them here (a simple cache)
_cache = {}


def _load():
    """Load test data, the Isolation Forest and the SHAP explainer (first call only)."""
    if not _cache:
        print("Loading data for SHAP (first time only, please wait)...")
        _, X_test, _, test_df, _ = preprocess()
        iso = joblib.load("models/isolation_forest.joblib")
        _cache["X_test"] = X_test
        _cache["test_df"] = test_df
        _cache["explainer"] = shap.TreeExplainer(iso)   # fast explainer for tree models
    return _cache


def _describe(name, scaled_value, raw_row):
    """Turn a column name into something a human can read.
    One-hot columns like 'service_private' become 'service = private'."""
    for col in CATEGORICAL_COLUMNS:
        if name.startswith(col + "_"):
            category = name[len(col) + 1:]
            return f"{col} = {category}", ("yes" if scaled_value > 0.5 else "no")

    # Normal numeric column: show the ORIGINAL value (not the scaled one)
    value = float(raw_row[name])
    value = int(value) if value.is_integer() else round(value, 3)
    return name, value


def explain_row(row_index, top_n=8):
    """Explain one test-set row. Returns the top_n most influential features."""
    c = _load()
    row = c["X_test"].iloc[[row_index]]          # the row as the model sees it
    raw_row = c["test_df"].iloc[row_index]       # the same row with original values

    # One SHAP number per feature (122 numbers for this row)
    shap_values = c["explainer"].shap_values(row.values)[0]

    # Pick the features with the biggest influence (ignoring + or -)
    top = np.argsort(-np.abs(shap_values))[:top_n]

    features = []
    for j in top:
        name, value = _describe(row.columns[j], float(row.iloc[0, j]), raw_row)
        s = float(shap_values[j])
        features.append({
            "feature": name,
            "value": value,
            "shap": round(s, 5),
            # negative = lower "normality" score = pushes toward attack
            "direction": "toward attack" if s < 0 else "toward normal",
        })

    return {
        "model": "Isolation Forest",
        "true_label": str(raw_row["attack_name"]),
        "features": features,
    }