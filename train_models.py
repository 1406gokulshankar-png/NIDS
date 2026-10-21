# train_models.py
# Purpose: train 3 models, compare them on the test set, save the best one.
#   1. Isolation Forest  (anomaly detector, learns from NORMAL traffic only)
#   2. Random Forest     (supervised, learns from normal + attack examples)
#   3. One-Class SVM     (anomaly detector, learns from NORMAL traffic only)

import os
import time
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from preprocess import preprocess        # our Phase 2 pipeline
from attack_mapping import CATEGORIES

os.makedirs("models", exist_ok=True)


def evaluate(name, y_true, y_pred):
    """Calculate accuracy, precision, recall, FPR, F1 for one model."""
    # confusion_matrix gives the four boxes: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred) * 100,
        "false_positive_rate": fp / (fp + tn) * 100,   # normal rows wrongly flagged
        "f1": f1_score(y_true, y_pred) * 100,
    }


def anomaly_to_label(model, X):
    """Anomaly detectors answer -1 (weird) or 1 (normal).
    We convert that to our style: 1 = attack, 0 = normal."""
    return (model.predict(X) == -1).astype(int)


# ------------------------------------------------------------------
# Step A: get the prepared data from Phase 2
# ------------------------------------------------------------------
print("Loading and preprocessing data...")
X_train, X_test, train_df, test_df, preprocessor = preprocess()

y_train = train_df["is_attack"]     # 0 = normal, 1 = attack
y_test = test_df["is_attack"]

# Normal-only rows from the train set (used by the 2 anomaly detectors)
X_train_normal = X_train[y_train == 0]
print("Train rows:", len(X_train), "| Normal-only rows:", len(X_train_normal))
print("Test rows :", len(X_test))

results = []        # one row of metrics per model
predictions = {}    # keep each model's predictions for the category table
trained = {}        # keep each trained model so we can save it

# ------------------------------------------------------------------
# Model 1: Isolation Forest
# ------------------------------------------------------------------
print("\nTraining Isolation Forest...")
start = time.time()
iso = IsolationForest(
    n_estimators=100,
    contamination=0.05,   # "flag the weirdest 5% of normal traffic" (the threshold knob)
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_train_normal)                       # learns ONLY from normal traffic
pred = anomaly_to_label(iso, X_test)
results.append(evaluate("Isolation Forest", y_test, pred))
predictions["Isolation Forest"] = pred
trained["Isolation Forest"] = ("anomaly", iso)
print(f"  done in {time.time() - start:.1f} seconds")

# ------------------------------------------------------------------
# Model 2: Random Forest (supervised)
# ------------------------------------------------------------------
print("\nTraining Random Forest...")
start = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)                      # learns from normal AND attack examples
pred = rf.predict(X_test)
results.append(evaluate("Random Forest", y_test, pred))
predictions["Random Forest"] = pred
trained["Random Forest"] = ("supervised", rf)
print(f"  done in {time.time() - start:.1f} seconds")

# ------------------------------------------------------------------
# Model 3: One-Class SVM (slow on big data, so we use 10,000 normal rows)
# ------------------------------------------------------------------
print("\nTraining One-Class SVM (this one is slower, be patient)...")
start = time.time()
X_small = X_train_normal.sample(n=10000, random_state=42)
svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
svm.fit(X_small)
pred = anomaly_to_label(svm, X_test)
results.append(evaluate("One-Class SVM", y_test, pred))
predictions["One-Class SVM"] = pred
trained["One-Class SVM"] = ("anomaly", svm)
print(f"  done in {time.time() - start:.1f} seconds")

# ------------------------------------------------------------------
# Comparison table
# ------------------------------------------------------------------
results_df = pd.DataFrame(results).set_index("model").round(2)

print("\n" + "=" * 70)
print("MODEL COMPARISON (all numbers are %, attack = positive class)")
print("=" * 70)
print(results_df)
print("\nAccuracy: overall correct | Precision: alarms that were real")
print("Recall: attacks caught    | False-positive rate: normal flagged (lower = better)")

# ------------------------------------------------------------------
# Detection by category: % of each traffic type flagged as "attack"
# ------------------------------------------------------------------
groups = {cat: (test_df["category"] == cat).values for cat in CATEGORIES}
groups["Novel (unseen)"] = test_df["is_novel"].values

rows = {}
for model_name, pred in predictions.items():
    pred_series = pd.Series(pred)
    rows[model_name] = {g: pred_series[mask].mean() * 100 for g, mask in groups.items()}
category_df = pd.DataFrame(rows).round(1)

print("\n" + "=" * 70)
print("% OF EACH TRAFFIC TYPE FLAGGED AS ATTACK")
print("  Normal row  = false alarm rate (want LOW)")
print("  Other rows  = detection rate   (want HIGH)")
print("=" * 70)
print(category_df)

# ------------------------------------------------------------------
# Pick the best model by F1 score, then save everything
# ------------------------------------------------------------------
best_name = results_df["f1"].idxmax()
best_kind, best_model = trained[best_name]
print("\nBest model (highest F1):", best_name)

# Save the best model together with a little info about it
joblib.dump(
    {"name": best_name, "kind": best_kind, "model": best_model},
    "models/best_model.joblib",
    compress=3,
)

# Also save the Isolation Forest separately: later it becomes our
# "unknown / possible zero-day" detector, whichever model wins here.
joblib.dump(iso, "models/isolation_forest.joblib", compress=3)

# Save the comparison table so we can quote it in the report
results_df.to_csv("models/model_comparison.csv")

print("Saved: models/best_model.joblib")
print("Saved: models/isolation_forest.joblib")
print("Saved: models/model_comparison.csv")