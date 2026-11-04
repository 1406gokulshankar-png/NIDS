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
from database import db, Alert, ReplayLog          # NEW: ReplayLog added
from severity import compute_severity
from evaluation import evaluate, print_summary, save_evaluation   # NEW

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

# ---- Get the data (preprocess() returns 5 things, we need three) ----
print("Loading test data...")
X_train, X_test, _, test_df, _ = preprocess()

# Pick random rows from the test set (the same seed gives the same rows)
rng = np.random.RandomState(args.seed)
n = min(args.rows, len(X_test))
row_numbers = rng.choice(len(X_test), size=n, replace=False)


def make_scaler(model):
    """Build a function that rescales a model's raw anomaly value to 0.5 - 1.0.
    0.5 = right at the model's alarm threshold.
    1.0 = as strange as the top 20% of the rows the model flags."""
    ref = X_train.sample(n=5000, random_state=0)       # a sample is enough
    raw_ref = -model.score_samples(ref)                # higher = stranger
    threshold = -model.offset_                         # the model's alarm line

    # Look only at rows the model would flag, and use their spread as the scale
    flagged_ref = raw_ref[raw_ref > threshold]
    if len(flagged_ref) > 0:
        top = max(np.percentile(flagged_ref, 80), threshold + 1e-6)
    else:
        top = threshold + 1e-6

    def scale(raw):
        s = (raw - threshold) / (top - threshold)      # 0 at threshold, 1 at "top"
        return float(0.5 + 0.5 * min(max(s, 0.0), 1.0))
    return scale


iso_scale = make_scaler(iso)
best_scale = make_scaler(best_model) if best_kind == "anomaly" else None


def check_row(row):
    """Return (attack_type, score) if the row is suspicious, else None."""
    # --- Layer 1: the best model from Phase 3 ---
    if best_kind == "supervised":
        score = float(best_model.predict_proba(row)[0][1])   # chance it is an attack
        flagged = score >= 0.5
    else:  # an anomaly detector won
        flagged = best_model.predict(row)[0] == -1
        score = best_scale(-best_model.score_samples(row)[0])

    if flagged:
        category = cat_model.predict(row)[0]
        if category == "Normal":          # the two models disagree
            category = "Unknown"
        return category, score

    # --- Layer 2: Isolation Forest looks for anything strange ---
    if iso.predict(row)[0] == -1:
        return "Unknown", iso_scale(-iso.score_samples(row)[0])

    return None                            # looks normal, no alert


# ---- The replay loop ----
with app.app_context():
    if args.clear:
        Alert.query.delete()
        ReplayLog.query.delete()           # NEW: also clear the replay log
        db.session.commit()
        print("Old alerts deleted.")

    print(f"\nReplaying {n} rows...\n")
    alerts_made = 0
    caught = 0          # real attacks that raised an alert
    false_alarms = 0    # normal rows that raised an alert
    severity_counts = Counter()   # how many alerts of each severity
    results = []        # NEW: one entry for EVERY replayed row (used for evaluation)

    for i in row_numbers:
        row = X_test.iloc[[int(i)]]            # one row, still with column names
        info = test_df.iloc[int(i)]            # original info about that row
        result = check_row(row)

        # ---- NEW (Feature 1): remember what happened to this row, alert or not ----
        flagged = result is not None
        results.append({
            "row_index": int(i),
            "attack_name": info["attack_name"],             # true name, e.g. "neptune"
            "true_category": info["category"],              # true category, e.g. "DoS"
            "is_novel": bool(info["is_novel"]),             # True = never seen in training
            "flagged": flagged,                             # did an alert fire?
            "alert_type": result[0] if flagged else None,   # e.g. "DoS" or "Unknown"
        })

        # ---- NEW (Feature 2): log EVERY row for the drift monitor ----
        db.session.add(ReplayLog(test_row_index=int(i), flagged=flagged))
        if len(results) % 100 == 0:        # save in batches of 100 (faster than every row)
            db.session.commit()

        if result is not None:
            attack_type, score = result
            severity = compute_severity(attack_type, score)
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

    db.session.commit()                        # NEW: save any log rows still waiting

    total_attacks = int(test_df.iloc[row_numbers]["is_attack"].sum())
    print("\n" + "=" * 50)
    print("REPLAY SUMMARY")
    print("=" * 50)
    print("Rows replayed        :", n)
    print("Alerts raised        :", alerts_made)                       # NEW: print what we counted
    print(f"Attacks caught       : {caught} of {total_attacks}")        # NEW
    print("False alarms         :", false_alarms)                      # NEW
    print("Alerts by severity   :", {lvl: severity_counts[lvl]         # NEW
                                     for lvl in ("Critical", "High", "Medium", "Low")})

    # ---- NEW (Feature 1): novel-attack evaluation ----
    metrics = evaluate(results)
    print_summary(metrics)
    save_evaluation(metrics)
    print("Saved evaluation.json")