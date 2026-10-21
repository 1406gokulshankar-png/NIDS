# explore_data.py
# Purpose: look at the dataset (size, columns, labels, missing values).

import pandas as pd
from data_loader import load_train_test   # reuse the loader we just wrote

# Make pandas show all columns instead of hiding the middle ones with "..."
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Load both tables
train_df, test_df = load_train_test()

# ---------- 1. Size of the data ----------
print("=" * 60)
print("1. DATASET SIZE (rows, columns)")
print("=" * 60)
print("Train:", train_df.shape)
print("Test :", test_df.shape)

# ---------- 2. Column names and data types ----------
print("\n" + "=" * 60)
print("2. COLUMN NAMES AND DATA TYPES")
print("=" * 60)
# Numbers show as int64/float64. Text columns show as 'str' (or 'object').
print(train_df.dtypes.to_string())

# ---------- 3. First few rows ----------
print("\n" + "=" * 60)
print("3. FIRST 5 ROWS")
print("=" * 60)
print(train_df.head())

# ---------- 4. Label counts ----------
# 'label' tells us if a connection is normal or which attack it is
print("\n" + "=" * 60)
print("4. LABEL COUNTS (TRAIN)")
print("=" * 60)
print(train_df["label"].value_counts().to_string())

print("\nDifferent labels in train:", train_df["label"].nunique())
print("Different labels in test :", test_df["label"].nunique())

# Simple view: normal vs attack
is_normal = train_df["label"] == "normal"
print("\nNormal vs Attack (TRAIN):")
print("Normal:", is_normal.sum())
print("Attack:", (~is_normal).sum())

# Attacks that appear in the test file but NOT in the train file
new_attacks = set(test_df["label"]) - set(train_df["label"])
print("\nAttack types ONLY in the test set (model never studies these):")
print(sorted(new_attacks))

# ---------- 5. Missing values ----------
print("\n" + "=" * 60)
print("5. MISSING VALUES")
print("=" * 60)
print("Total missing values in train:", train_df.isnull().sum().sum())
print("Total missing values in test :", test_df.isnull().sum().sum())

# ---------- 6. Text (categorical) columns ----------
print("\n" + "=" * 60)
print("6. TEXT COLUMNS (we must convert these to numbers later)")
print("=" * 60)
for col in ["protocol_type", "service", "flag"]:
    print(f"{col}: {train_df[col].nunique()} unique values")
print("\nprotocol_type values:", train_df["protocol_type"].unique().tolist())