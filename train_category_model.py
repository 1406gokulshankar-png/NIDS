# train_category_model.py
# Purpose: train a small model that says WHICH category an attack belongs to
# (Normal, DoS, Probe, R2L, U2R). The replay script uses it to label alerts.

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from preprocess import preprocess     # our Phase 2 pipeline

print("Loading and preprocessing data...")
X_train, X_test, train_df, test_df, preprocessor = preprocess()

# This time the answer column is the 5-way category, not just 0/1
y_train = train_df["category"]
y_test = test_df["category"]

print("Training category model (about 15-30 seconds)...")
cat_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cat_model.fit(X_train, y_train)

# Quick check on the test set
pred = cat_model.predict(X_test)
print("\nCategory model results on the test set:")
print(classification_report(y_test, pred, zero_division=0))

joblib.dump(cat_model, "models/category_model.joblib", compress=3)
print("Saved: models/category_model.joblib")