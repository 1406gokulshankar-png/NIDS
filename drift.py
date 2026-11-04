# drift.py
# Purpose: concept drift monitor.
# It asks one question: "Does the traffic we are seeing now still look like the
# traffic our models were trained on?" If not, the models may be getting stale.

import numpy as np
from scipy.stats import ks_2samp

# ---------------- Settings you can change ----------------
PSI_MODERATE = 0.10    # PSI at or above this = a small shift
PSI_MAJOR = 0.25       # PSI at or above this = a big shift (we say the feature "drifted")
WARNING_SHARE = 0.10   # Warning if this share of features (or more) drifted
STALE_SHARE = 0.25     # Stale   if this share of features (or more) drifted
MIN_ROWS = 100         # fewer replayed rows than this = too few to judge
TRAIN_SAMPLE = 20000   # how many training rows we compare against (keeps it fast)
TOP_N = 5              # how many top drifting features to report
EPSILON = 1e-4         # stops us from taking log(0) in the PSI formula
# ---------------------------------------------------------


def make_bin_edges(train_values):
    """Decide where to cut the values into bins, using the TRAINING data."""
    train_values = np.asarray(train_values, dtype=float)

    # Cut points at the 10%, 20% ... 90% marks of the training values
    cuts = set(np.quantile(train_values, np.linspace(0.1, 0.9, 9)).tolist())

    # Many columns are mostly one value (for example 0). Give that most common
    # value its own bin, so a sudden rise in non-zero values is noticed.
    values, counts = np.unique(train_values, return_counts=True)
    most_common = values[np.argmax(counts)]
    cuts.add(float(most_common))
    cuts.add(float(np.nextafter(most_common, np.inf)))   # the tiniest step above it

    return np.array(sorted(cuts))


def bin_proportions(values, edges):
    """What share of the values falls into each bin?"""
    bin_numbers = np.searchsorted(edges, values, side="right")
    counts = np.bincount(bin_numbers, minlength=len(edges) + 1)
    return counts / len(values)


def psi(train_values, new_values):
    """Population Stability Index. 0 means the two sets are spread identically."""
    train_values = np.asarray(train_values, dtype=float)
    new_values = np.asarray(new_values, dtype=float)

    edges = make_bin_edges(train_values)          # bins come from the training data
    expected = np.clip(bin_proportions(train_values, edges), EPSILON, None)
    actual = np.clip(bin_proportions(new_values, edges), EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def ks_statistic(train_values, new_values):
    """KS test. We keep the statistic (0 to 1) and ignore the p-value (see README)."""
    return float(ks_2samp(train_values, new_values).statistic)


def psi_level(value):
    """Turn a PSI number into a word."""
    if value >= PSI_MAJOR:
        return "major"
    if value >= PSI_MODERATE:
        return "moderate"
    return "low"


def status_from_share(share):
    """share = fraction of features that drifted (0.0 to 1.0)."""
    if share >= STALE_SHARE:
        return "Stale"
    if share >= WARNING_SHARE:
        return "Warning"
    return "Stable"


MESSAGES = {
    "Stable": "Traffic still looks like the training data.",
    "Warning": "Some features have shifted. Keep an eye on detection results.",
    "Stale": "Traffic looks quite different from the training data. Consider retraining.",
}


def analyze(train_columns, new_columns):
    """Compare two sets of columns and return the drift report.

    Both arguments are dictionaries:  {feature name: numpy array of values}
    """
    rows = len(next(iter(new_columns.values())))
    if rows < MIN_ROWS:
        return {
            "status": "Not enough data",
            "rows_checked": rows,
            "features_checked": 0,
            "features_drifted": 0,
            "drift_share_pct": 0.0,
            "top_features": [],
            "message": f"Need at least {MIN_ROWS} replayed rows (have {rows}).",
        }

    # Score every feature
    scores = []
    for name, train_values in train_columns.items():
        new_values = new_columns[name]
        p = psi(train_values, new_values)
        scores.append({
            "feature": name,
            "psi": round(p, 3),
            "ks": round(ks_statistic(train_values, new_values), 3),
            "level": psi_level(p),
        })

    drifted = [s for s in scores if s["level"] == "major"]
    share = len(drifted) / len(scores)
    status = status_from_share(share)

    # Most drifting features first
    scores.sort(key=lambda s: (s["psi"], s["ks"]), reverse=True)

    return {
        "status": status,
        "rows_checked": rows,
        "features_checked": len(scores),
        "features_drifted": len(drifted),
        "drift_share_pct": round(share * 100, 1),
        "top_features": scores[:TOP_N],
        "message": MESSAGES[status],
        "thresholds": {
            "psi_major": PSI_MAJOR,
            "warning_share_pct": WARNING_SHARE * 100,
            "stale_share_pct": STALE_SHARE * 100,
        },
    }


# ----------------------------------------------------------------------
# Loading the real data (slow the first time, then remembered)
# ----------------------------------------------------------------------
_reference = None


def get_reference():
    """Load and prepare the training data once, then reuse it."""
    global _reference
    if _reference is None:
        # Imported here so that importing drift.py stays fast
        from preprocess import preprocess, NUMERIC_COLUMNS

        X_train, X_test, _, test_df, _ = preprocess()

        # A random sample of the training rows is enough, and much faster
        sample = X_train.sample(n=min(TRAIN_SAMPLE, len(X_train)), random_state=0)
        # Fresh training rows NOT in the sample (used by the self-test)
        rest = X_train.drop(sample.index)
        holdout = rest.sample(n=min(2000, len(rest)), random_state=1)

        _reference = {
            "columns": NUMERIC_COLUMNS,
            "train": {c: sample[c].to_numpy() for c in NUMERIC_COLUMNS},
            "holdout": holdout[NUMERIC_COLUMNS],
            "test": X_test[NUMERIC_COLUMNS],
            "novel_rows": np.where(test_df["is_novel"].to_numpy())[0],
        }
    return _reference


def compute_drift(row_indices):
    """Drift report for the test rows that were replayed (list of row numbers)."""
    ref = get_reference()
    replayed = ref["test"].iloc[row_indices]
    new_columns = {c: replayed[c].to_numpy() for c in ref["columns"]}
    return analyze(ref["train"], new_columns)


def print_result(title, result):
    """Print a drift report in a readable way."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print("Status           :", result["status"])
    print("Rows checked     :", result["rows_checked"])
    print("Features drifted :", result["features_drifted"], "of", result["features_checked"],
          f"({result['drift_share_pct']}%)")
    print(result["message"])
    for f in result["top_features"]:
        print(f"  {f['feature']:<30} PSI {f['psi']:<7} KS {f['ks']:<6} {f['level']}")


# ----------------------------------------------------------------------
# Command line:
#   python drift.py              -> check all rows the replay has logged
#   python drift.py --selftest   -> check the monitor against known situations
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        ref = get_reference()

        # Situation 1: fresh rows from the SAME training data. Should be Stable.
        same = {c: ref["holdout"][c].to_numpy() for c in ref["columns"]}
        print_result("SELF-TEST 1: new rows from the TRAINING data (expect Stable)",
                     analyze(ref["train"], same))

        # Situation 2: only attacks the model never saw. Should NOT be Stable.
        novel = ref["test"].iloc[ref["novel_rows"]]
        novel_columns = {c: novel[c].to_numpy() for c in ref["columns"]}
        print_result("SELF-TEST 2: only NOVEL attack rows (expect Warning or Stale)",
                     analyze(ref["train"], novel_columns))
    else:
        from app import app
        from database import ReplayLog

        with app.app_context():
            indices = [log.test_row_index for log in ReplayLog.query.all()]
        print_result("DRIFT CHECK on all logged replay rows", compute_drift(indices))