from pathlib import Path

import pandas as pd


REDTEAM_FILE = Path("raw/redteam.txt.gz")
OUTPUT_FILE = Path("outputs/tables/redteam_window_labels.parquet")

WINDOW_SECONDS = 300

REDTEAM_COLUMNS = [
    "time",
    "src_user",
    "src_comp",
    "dst_comp"
]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

redteam = pd.read_csv(
    REDTEAM_FILE,
    header=None,
    names=REDTEAM_COLUMNS,
    compression="gzip",
)

redteam["window_id"] = redteam["time"] // WINDOW_SECONDS
redteam["day"] = redteam["time"] // 86400

# window-level labels
labels = (
    redteam
    .groupby("window_id", as_index=False)
    .agg(
        day=("day", "first"),
        redteam_event_count=("time", "size"),
        redteam_source_users=("src_user", lambda x: sorted(set(x))),
        redteam_source_computers=("src_comp", lambda x: sorted(set(x))),
        redteam_destination_computers=("dst_comp", lambda x: sorted(set(x))),
    )
)

labels["redteam_flag"] = 1

labels = labels[
    [
        "window_id",
        "day",
        "redteam_flag",
        "redteam_event_count",
        "redteam_source_users",
        "redteam_source_computers",
        "redteam_destination_computers",
    ]
]

labels.to_parquet(OUTPUT_FILE, index=False)
