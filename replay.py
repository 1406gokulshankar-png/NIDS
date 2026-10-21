# replay.py
# Purpose: simulate live network traffic.
# It takes test-set rows one by one, asks the models if each row is suspicious,
# and saves every flagged row as an Alert in the database.

import os
import sys
import time
import argparse
import joblib
import numpy as np
from collections import Counter
from preprocess import preprocess
from app import app                      # our Flask app (gives access to the database)
from database import db, Alert
from severity import compute_severity

# ---- Settings you can change from the command line ----
parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=300, help="how many test rows to replay")
parser.add_argument("--delay", type=float, default=0.05, help="pause between rows (seconds)")
parser.add_argument("--seed", type=int, default=1, help="random seed for picking rows")
parser.add_argument("--clear", action="store_true", help="delete old alerts first")
args = parser.parse_args()

# ---- Make sure the saved models from earlier phases exist ----
needed = [
    "models/best_model.joblib",
    "models/isolation_forest.joblib",
    "models/category_model.joblib",
]
missing = [p for p in needed if not os.path.exists(p)]
if missing:
    print("These files are missing:", missing)
    print("Run python train_models.py and python train_category_model.py first.")
    sys.exit(1)

# ---- Load the models ----
saved = joblib.load("models/best_model.joblib")
best_name, best_kind, best_model = saved["name"], saved["kind"], saved["model"]
iso = joblib.load("models/isolation_forest.joblib")
cat_model = joblib.load("models/category_model.joblib")

# Predicting one row at a time is faster without multi-threading
for m in (best_model, iso, cat_model):
    if hasattr(m, "n_jobs"):
        m.n_jobs = 1

print("Layer 1 model:", best_name, "| Layer 2: Isolation Forest")

# ---- Get the test data (preprocess() returns 5 things, we need two) ----
print("Loading test data...")
_, X_test, _, test_df, _ = preprocess()

# Pick random rows from the test set (the same seed gives the same rows)
rng = np.random.RandomState(args.seed)
n = min(args.rows, len(X_test))
row_numbers = rng.choice(len(X_test), size=n, replace=False)


def check_row(row):
    """Return (attack_type, score) if the row is suspicious, else None."""
    # --- Layer 1: the best model from Phase 3 ---
    if best_kind == "supervised":
        score = float(best_model.predict_proba(row)[0][1])   # chance it is an attack
        flagged = score >= 0.5
    else:  # an anomaly detector won
        flagged = best_model.predict(row)[0] == -1
        score = float(-best_model.decision_function(row)[0])

    if flagged:
        category = cat_model.predict(row)[0]
        if category == "Normal":          # the two models disagree
            category = "Unknown"
        return category, score

    # --- Layer 2: Isolation Forest looks for anything strange ---
    if iso.predict(row)[0] == -1:
        return "Unknown", float(-iso.score_samples(row)[0])

    return None                            # looks normal, no alert


# ---- The replay loop ----
with app.app_context():
    if args.clear:
        Alert.query.delete()
        db.session.commit()
        print("Old alerts deleted.")

    print(f"\nReplaying {n} rows...\n")
    alerts_made = 0
    caught = 0          # real attacks that raised an alert
    false_alarms = 0    # normal rows that raised an alert
    severity_counts = Counter()   # how many alerts of each severity

    for i in row_numbers:
        row = X_test.iloc[[int(i)]]            # one row, still with column names
        info = test_df.iloc[int(i)]            # original info about that row
        result = check_row(row)

        if result is not None:
            attack_type, score = result
            severity = compute_severity(attack_type, score)   # NEW in Phase 6
            alert = Alert(
                source_info=f"{info['protocol_type']} / {info['service']}",
                attack_type=attack_type,
                anomaly_score=score,
                severity=severity,
                test_row_index=int(i),
            )
            db.session.add(alert)
            db.session.commit()
            alerts_made += 1
            severity_counts[severity] += 1
            if info["is_attack"] == 1:
                caught += 1
            else:
                false_alarms += 1
            print(f"ALERT #{alert.id:<4} {attack_type:<8} {severity:<9} "
                  f"score={score:.3f}  (truth: {info['attack_name']})")

        time.sleep(args.delay)                 # pretend traffic arrives over time

    total_attacks = int(test_df.iloc[row_numbers]["is_attack"].sum())
    print("\n" + "=" * 50)
    print("REPLAY SUMMARY")
    print("=" * 50)
    print("Rows replayed        :", n)
    print("Alerts saved         :", alerts_made)
    print("Real attacks in rows :", total_attacks)
    print("Attacks caught       :", caught)
    print("False alarms         :", false_alarms)
    print("\nAlerts by severity:")
    for level in ["Low", "Medium", "High", "Critical"]:
        print(f"  {level:<9}: {severity_counts[level]}")