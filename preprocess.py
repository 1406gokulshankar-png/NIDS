# preprocess.py
# Purpose: prepare the data for machine learning.
#   1. Add category labels (Normal / DoS / Probe / R2L / U2R)
#   2. Convert text columns to numbers (one-hot encoding)
#   3. Scale the number columns (StandardScaler)
#   4. Save the fitted preprocessor so we can reuse it later

import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import load_train_test, FEATURE_COLUMNS
from attack_mapping import ATTACK_CATEGORY, CATEGORIES

# The 3 columns that contain text
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# All the other feature columns contain numbers
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

# Where we will save the fitted preprocessor
PREPROCESSOR_PATH = "models/preprocessor.joblib"


def add_label_columns(df):
    """Add useful label columns to a table and return the new table."""
    df = df.copy()  # work on a copy so the original table stays untouched

    # Keep the original attack name (e.g. "neptune")
    df["attack_name"] = df["label"]

    # Look up the category (e.g. "neptune" -> "DoS")
    df["category"] = df["label"].map(ATTACK_CATEGORY)

    # If any label is missing from our dictionary, stop and tell us which
    unmapped = df.loc[df["category"].isna(), "label"].unique()
    if len(unmapped) > 0:
        raise ValueError(f"These labels are not in ATTACK_CATEGORY: {list(unmapped)}")

    # Simple yes/no version: 0 = normal, 1 = any attack
    df["is_attack"] = (df["category"] != "Normal").astype(int)
    return df


def build_preprocessor():
    """Create the (not yet trained) preprocessor."""
    preprocessor = ColumnTransformer(
        transformers=[
            # Text columns -> 0/1 columns.
            # handle_unknown="ignore": if a new, unseen value appears later,
            # give it all zeros instead of crashing.
            ("cat",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLUMNS),

            # Number columns -> scaled so average is ~0 and spread is ~1
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ],
        verbose_feature_names_out=False,  # keep column names short and clean
    )
    # Make the output a pandas table with column names (not a bare array)
    preprocessor.set_output(transform="pandas")
    return preprocessor


def preprocess():
    """Load data, add labels, fit the preprocessor on TRAIN, transform both.

    Returns: X_train, X_test, train_df, test_df, preprocessor
      - X_train / X_test : numeric tables the model can use
      - train_df / test_df : original tables with label columns
    """
    train_df, test_df = load_train_test()
    train_df = add_label_columns(train_df)
    test_df = add_label_columns(test_df)

    # --- Mark "novel" attacks: attacks in the test set never seen in training ---
    # (We will use this in our "Unknown / Possible Zero-Day" feature.)
    train_attack_names = set(train_df["attack_name"])
    train_df["is_novel"] = False
    test_df["is_novel"] = ~test_df["attack_name"].isin(train_attack_names)

    preprocessor = build_preprocessor()

    # fit_transform = LEARN the rules from train, then apply them
    X_train = preprocessor.fit_transform(train_df[FEATURE_COLUMNS])
    # transform = only APPLY the rules learned from train (no learning!)
    X_test = preprocessor.transform(test_df[FEATURE_COLUMNS])

    return X_train, X_test, train_df, test_df, preprocessor


# This block runs only when we type:  python preprocess.py
if __name__ == "__main__":
    X_train, X_test, train_df, test_df, preprocessor = preprocess()

    print("=" * 60)
    print("1. SHAPE BEFORE vs AFTER PREPROCESSING")
    print("=" * 60)
    print("Before (train features):", train_df[FEATURE_COLUMNS].shape)
    print("After  (train features):", X_train.shape)
    print("After  (test features) :", X_test.shape)

    print("\n" + "=" * 60)
    print("2. SOME OF THE NEW COLUMN NAMES")
    print("=" * 60)
    print("First 6:", list(X_train.columns[:6]))
    print("Last 3 :", list(X_train.columns[-3:]))

    print("\n" + "=" * 60)
    print("3. CATEGORY COUNTS")
    print("=" * 60)
    counts = pd.DataFrame({
        "train": train_df["category"].value_counts(),
        "test": test_df["category"].value_counts(),
    }).reindex(CATEGORIES)
    counts["train_%"] = (counts["train"] / len(train_df) * 100).round(1)
    counts["test_%"] = (counts["test"] / len(test_df) * 100).round(1)
    print(counts)

    print("\n" + "=" * 60)
    print("4. NOVEL ATTACKS (in test set only)")
    print("=" * 60)
    novel_count = test_df["is_novel"].sum()
    print("Test rows from attacks never seen in training:", novel_count)
    print("That is", round(novel_count / len(test_df) * 100, 1), "% of the test set")

    print("\n" + "=" * 60)
    print("5. SCALING CHECK (train columns should have mean ~0, std ~1)")
    print("=" * 60)
    print(X_train[["duration", "src_bytes", "count"]].agg(["mean", "std"]).round(3))

    print("\n" + "=" * 60)
    print("6. MISSING VALUES AFTER PREPROCESSING")
    print("=" * 60)
    print("Train:", X_train.isnull().sum().sum(), " Test:", X_test.isnull().sum().sum())

    # Save the fitted preprocessor for later phases
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("\nSaved preprocessor to:", PREPROCESSOR_PATH)