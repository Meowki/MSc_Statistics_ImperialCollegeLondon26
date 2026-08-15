from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS_FILE = Path(
    "outputs/tables/user_window_counts.parquet"
)
REDTEAM_FILE = Path(
    "outputs/tables/redteam_user_window_labels.parquet"
)
SCORES_FILE = Path(
    "outputs/tables/25_human_backsolved_scores.parquet"
)

OUT_ALL = Path(
    "outputs/tables/29_redteam_eligibility.csv"
)
OUT_MISSING = Path(
    "outputs/tables/29_redteam_excluded.csv"
)

HISTORY_WINDOWS = 7 * 288

OUT_ALL.parent.mkdir(parents=True, exist_ok=True)


labels = pd.read_parquet(
    REDTEAM_FILE,
    columns=["src_user", "window_id"],
)

scores = pd.read_parquet(
    SCORES_FILE,
    columns=["src_user", "window_id"],
)
scores["eligible_25"] = 1

labels = labels.merge(
    scores,
    on=["src_user", "window_id"],
    how="left",
)
labels["eligible_25"] = (
    labels["eligible_25"]
    .fillna(0)
    .astype(int)
)


# active windows for red-team users
features = pd.read_parquet(
    WINDOWS_FILE,
    columns=[
        "src_user",
        "window_id",
    ],
    filters=[("account_type", "==", "human")],
)

features = features[
    features["src_user"].isin(labels["src_user"])
].sort_values(["src_user", "window_id"])


rows = []

for user, user_labels in labels.groupby("src_user"):
    windows = features.loc[
        features["src_user"] == user,
        "window_id",
    ].to_numpy()

    for row in user_labels.itertuples():
        end = np.searchsorted(
            windows,
            row.window_id,
            side="left",
        )
        start = np.searchsorted(
            windows,
            row.window_id - HISTORY_WINDOWS,
            side="left",
        )

        rows.append({
            "src_user": user,
            "window_id": row.window_id,
            "eligible_25": row.eligible_25,
            "prior_active_all": end,
            "prior_active_7d": end - start,
            "first_active_observation": end == 0,
        })


result = pd.DataFrame(rows)

result["exclusion_reason"] = ""
result.loc[
    (result["eligible_25"] == 0)
    & result["first_active_observation"],
    "exclusion_reason",
] = "first active observation"

result.loc[
    (result["eligible_25"] == 0)
    & ~result["first_active_observation"]
    & (result["prior_active_7d"] == 0),
    "exclusion_reason",
] = "no active history in previous 7 days"

missing = result[result["eligible_25"] == 0]

result.to_csv(OUT_ALL, index=False)
missing.to_csv(OUT_MISSING, index=False)

print(f"total red-team rows: {len(result)}")
print(f"eligible windows: {result['eligible_25'].sum()}")
print(f"excluded from 25: {len(missing)}")
print()
print("excluded reasons:")
print(missing["exclusion_reason"].value_counts())
print()
print(missing.to_string(index=False))

print(f"\nsaved {OUT_ALL}")
print(f"saved {OUT_MISSING}")
