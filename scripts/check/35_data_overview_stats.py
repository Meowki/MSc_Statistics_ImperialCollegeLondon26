from pathlib import Path

import pandas as pd


WINDOWS_FILE = Path("outputs/tables/user_window_counts.parquet")
REDTEAM_FILE = Path("outputs/tables/redteam_user_window_labels.parquet")
WINDOWS_PER_DAY = 288


windows = pd.read_parquet(
    WINDOWS_FILE,
    columns=["src_user", "window_id", "account_type", "event_count"],
)
windows["day"] = windows["window_id"] // WINDOWS_PER_DAY

window_count = int(windows["window_id"].max()) + 1
day_count = int(windows["day"].max()) + 1

# overall scale
print("== overall ==")
print(f"active user-windows: {len(windows):,}")
print(f"total events:        {int(windows['event_count'].sum()):,}")
print(f"distinct users:      {windows['src_user'].nunique():,}")
print(f"windows / days:      {window_count:,} / {day_count}")

# account types
print("\n== by account type ==")
account_groups = windows.groupby("account_type")
account_summary = pd.DataFrame({
    "users": account_groups["src_user"].nunique(),
    "active_windows": account_groups.size(),
    "events": account_groups["event_count"].sum(),
})
account_summary["window_share"] = (
    account_summary["active_windows"] / len(windows)
).round(4)
print(account_summary)

# human active-window counts
human_counts = windows.loc[
    windows["account_type"] == "human",
    "event_count",
]
print("\n== event_count (human active) ==")
print(f"median count: {human_counts.median():.0f}")
print(f"maximum count: {human_counts.max():.0f}")

# weekly pattern
daily_totals = windows.groupby("day")["event_count"].sum()
weekday_means = daily_totals.groupby(daily_totals.index % 7).mean()
print("\n== weekly (all accounts, mean daily total by day-of-week, millions) ==")
print((weekday_means / 1e6).round(2))

# period means
print("\n== drift (mean daily total, millions) ==")
for name, start, end in [
    ("L 0-29", 0, 29),
    ("C 30-43", 30, 43),
    ("E 44-57", 44, 57),
]:
    print(f"{name}: {daily_totals.loc[start:end].mean() / 1e6:.2f}")
print(
    "E/C ratio: "
    f"{daily_totals.loc[44:57].mean() / daily_totals.loc[30:43].mean():.2f}"
)

# red-team labels
redteam = pd.read_parquet(REDTEAM_FILE)
print("\n== red-team ==")
print(f"labelled user-windows:   {len(redteam):,}")
print(f"distinct red-team users: {redteam['src_user'].nunique():,}")
print(f"n labelled days:         {redteam['day'].nunique()}")
print(f"labelled days:           {sorted(redteam['day'].unique())}")
