from pathlib import Path

import pandas as pd


SCORES_FILE = Path("outputs/tables/26_recalibrated_all.parquet")
OUTPUT_FILE = Path("outputs/tables/37_redteam_position.csv")
LABELLED_DAYS = (0, 29)


scores = pd.read_parquet(
    SCORES_FILE,
    columns=[
        "day",
        "event_count",
        "p_recal",
        "redteam_user_window_flag",
    ],
    filters=[
        ("account_type", "==", "human"),
        ("day", ">=", LABELLED_DAYS[0]),
        ("day", "<=", LABELLED_DAYS[1]),
    ],
)

# daily position among human windows
scores["daily_percentile"] = scores.groupby("day")["p_recal"].rank(
    method="average",
    ascending=True,
    pct=True,
)

rows = []
for group, flag in [("background", 0), ("redteam", 1)]:
    group_scores = scores[scores["redteam_user_window_flag"] == flag]
    counts = group_scores["event_count"]
    row = {
        "group": group,
        "n": len(group_scores),
        "count_mean": counts.mean(),
        "count_median": counts.median(),
        "count_p90": counts.quantile(0.90),
        "count_p99": counts.quantile(0.99),
        "count_max": counts.max(),
        "median_daily_pct": pd.NA,
        "share_top_1pct": pd.NA,
        "share_top_0p1pct": pd.NA,
    }

    if flag == 1:
        percentiles = group_scores["daily_percentile"]
        row["median_daily_pct"] = percentiles.median()
        row["share_top_1pct"] = (percentiles <= 0.01).mean()
        row["share_top_0p1pct"] = (percentiles <= 0.001).mean()

    rows.append(row)

pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nsaved {OUTPUT_FILE}")
