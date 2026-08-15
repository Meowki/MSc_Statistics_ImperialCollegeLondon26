from pathlib import Path

import pandas as pd


REDTEAM_WINDOWS_FILE = Path("outputs/tables/redteam_window_labels.parquet")
OUTPUT_FILE = Path("outputs/tables/redteam_user_window_labels.parquet")


labels = pd.read_parquet(
    REDTEAM_WINDOWS_FILE,
    columns=["window_id", "day", "redteam_source_users"],
)

# source-user window labels
user_window_labels = (
    labels
    .explode("redteam_source_users")
    .rename(columns={"redteam_source_users": "src_user"})
    [["src_user", "window_id", "day"]]
    .drop_duplicates()
)
user_window_labels["redteam_user_window_flag"] = 1

user_window_labels.to_parquet(OUTPUT_FILE, index=False)
