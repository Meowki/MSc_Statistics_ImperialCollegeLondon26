import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_utils import labelled_metrics

SCORES_FILE = Path("outputs/tables/26_recalibrated_all.parquet")
OUTPUT_FILE = Path("outputs/tables/22_policy_qchart_summary.csv")

GAMMA = 0.25
TAU_VALUES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

EVAL_DAYS = (44, 57)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


scores = pd.read_parquet(
    SCORES_FILE,
    columns=[
        "src_user",
        "day",
        "window_id",
        "account_type",
        "z",
        "redteam_user_window_flag",
    ],
)

# per-user smoothing
scores = scores.sort_values(
    ["account_type", "src_user", "window_id"]
).reset_index(drop=True)
scores["q"] = scores.groupby(
    ["account_type", "src_user"],
    sort=False,
)["z"].transform(
    lambda values: values.ewm(alpha=GAMMA, adjust=False).mean()
)


results = []
for tau in TAU_VALUES:
    for account_type in ["human", "machine"]:
        account_scores = scores[scores["account_type"] == account_type]
        alerts = account_scores[account_scores["q"] > tau]
        weak_label = labelled_metrics(alerts, account_type)
        evaluation_alerts = alerts[
            (alerts["day"] >= EVAL_DAYS[0])
            & (alerts["day"] <= EVAL_DAYS[1])
        ]
        daily_alerts = (
            evaluation_alerts.groupby("day")
            .size()
            .reindex(
                range(EVAL_DAYS[0], EVAL_DAYS[1] + 1),
                fill_value=0,
            )
        )
        mean_daily = float(daily_alerts.mean())
        sd_daily = float(daily_alerts.std())
        daily_cv = sd_daily / mean_daily if mean_daily > 0 else np.nan

        results.append({
            "gamma": GAMMA,
            "tau": tau,
            "account_type": account_type,
            **weak_label,
            "eval_daily_mean": mean_daily,
            "eval_daily_sd": sd_daily,
            "eval_daily_cv": daily_cv,
        })


summary = pd.DataFrame(results)
summary.to_csv(OUTPUT_FILE, index=False)
print("\nQ-chart EWMA summary:")
print(summary.to_string(index=False), "\n")

print(f"saved {OUTPUT_FILE}")
