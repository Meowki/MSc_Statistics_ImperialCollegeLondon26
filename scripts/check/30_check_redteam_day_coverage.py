from pathlib import Path

import pandas as pd

ELIGIBILITY_FILE = Path(
    "outputs/tables/29_redteam_eligibility.csv"
)
OUTPUT_FILE = Path(
    "outputs/tables/30_redteam_day_eligibility.csv"
)

WINDOWS_PER_DAY = 288

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


eligibility = pd.read_csv(ELIGIBILITY_FILE)
eligibility["day"] = eligibility["window_id"] // WINDOWS_PER_DAY


summary = (
    eligibility.groupby("day", as_index=False)
    .agg(
        total_labels=("window_id", "size"),
        eligible_labels=("eligible_25", "sum"),
    )
)

summary["excluded_labels"] = (
    summary["total_labels"]
    - summary["eligible_labels"]
)

summary["day_eligible"] = (
    summary["eligible_labels"] > 0
)

summary.to_csv(OUTPUT_FILE, index=False)


total_days = summary["day"].nunique()
eligible_days = int(summary["day_eligible"].sum())

print(f"total labelled days: {total_days}")
print(f"days with at least one eligible label: {eligible_days}")
print(
    f"days with no eligible labels: "
    f"{total_days - eligible_days}"
)

print("\nper-day eligibility:")
print(summary.to_string(index=False))
print(f"\nsaved {OUTPUT_FILE}")
