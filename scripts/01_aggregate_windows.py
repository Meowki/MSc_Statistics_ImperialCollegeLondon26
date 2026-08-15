from pathlib import Path

import pandas as pd


AUTH_FILE = Path("raw/auth.txt.gz")
OUTPUT_FILE = Path("outputs/tables/user_window_counts.parquet")

WINDOW_SECONDS = 300
CHUNK_SIZE = 1_000_000

AUTH_COLUMNS = [
    "time", "src_user", "dst_user", "src_comp", "dst_comp",
    "auth_type", "logon_type", "orientation", "outcome"
]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def classify_account(user):
    user = user.split("@")[0]

    if user.endswith("$") or user.startswith("C"):
        return "machine"
    if user.startswith("U"):
        return "human"
    if user.startswith("ANONYMOUS"):
        return "anonymous"
    return "other"


chunk_counts = []

for auth_chunk in pd.read_csv(
    AUTH_FILE,
    header=None,
    names=AUTH_COLUMNS,
    usecols=["time", "src_user", "outcome"],
    chunksize=CHUNK_SIZE,
    compression="gzip",
):
    auth_chunk["window_id"] = auth_chunk["time"] // WINDOW_SECONDS
    auth_chunk["is_failure"] = (
        auth_chunk["outcome"] == "Fail"
    ).astype("int8")

    # aggregation within each chunk
    counts = (
        auth_chunk
        .groupby(["src_user", "window_id"], as_index=False)
        .agg(
            event_count=("time", "size"),
            failure_count=("is_failure", "sum"),
        )
    )
    chunk_counts.append(counts)

# user-windows split across chunks
windows = (
    pd.concat(chunk_counts, ignore_index=True)
    .groupby(["src_user", "window_id"], as_index=False)
    .agg(
        event_count=("event_count", "sum"),
        failure_count=("failure_count", "sum"),
    )
)

# account types from LANL usernames
windows["account_type"] = windows["src_user"].map(classify_account)

windows = windows[
    [
        "src_user",
        "window_id",
        "account_type",
        "event_count",
        "failure_count",
    ]
]

windows.to_parquet(OUTPUT_FILE, index=False)
