# data_loader.py
# Purpose: read the NSL-KDD files and give every column a proper name.
# Other files in our project will import and reuse this file.

import pandas as pd

# ---- Where our dataset files are (matches your file names) ----
TRAIN_PATH = "data/KDDTrain+.txt"
TEST_PATH = "data/KDDTest+.txt"

# The dataset file has NO header row, so we write the column names ourselves.
# These 41 names describe each network connection, in the original order.
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]


def load_nsl_kdd(path):
    """Read one NSL-KDD file and return it as a table (DataFrame)."""

    # Look at the first line of the file to see if it has a header row
    with open(path, "r") as f:
        first_line = f.readline()
    has_header = first_line.lower().startswith("duration")

    if has_header:
        # The file already has column names, so pandas can use them
        df = pd.read_csv(path)
        if "class" in df.columns:
            df = df.rename(columns={"class": "label"})
    else:
        # No header: read the raw rows, then attach our own names
        df = pd.read_csv(path, header=None)
        if df.shape[1] == 43:
            # 41 features + label + difficulty score
            df.columns = FEATURE_COLUMNS + ["label", "difficulty"]
        elif df.shape[1] == 42:
            # 41 features + label only
            df.columns = FEATURE_COLUMNS + ["label"]
        else:
            raise ValueError(
                f"Expected 42 or 43 columns but found {df.shape[1]} in {path}"
            )

    # Clean the label text: lowercase, no spaces, no trailing dot
    # (so "Normal " and "normal." both become "normal")
    df["label"] = df["label"].astype(str).str.strip().str.lower().str.rstrip(".")
    return df


def load_train_test():
    """Load both files and return (train_table, test_table)."""
    train_df = load_nsl_kdd(TRAIN_PATH)
    test_df = load_nsl_kdd(TEST_PATH)
    return train_df, test_df