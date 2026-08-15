import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_utils import labelled_metrics

SCORES_FILE = Path("outputs/tables/26_recalibrated_all.parquet")
OUTPUT_FILE = Path("outputs/tables/21_policy_rate_summary.csv")

WINDOW_SEC = 300
WINDOWS_PER_DAY = 86400 // WINDOW_SEC
WINDOWS_PER_HOUR = 3600 // WINDOW_SEC
UPDATE_HOURS = 6
DELTA = UPDATE_HOURS * WINDOWS_PER_HOUR

LOOKBACK_HOURS_LIST = [24, 168]
TARGET_RATES = [100, 500]

EVAL_DAYS = (44, 57)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def controlled_rate(scores, target_rate, lookback_windows):
    window_ids = scores["window_id"].to_numpy()
    p_values = scores["p_recal"].to_numpy()
    first_window = window_ids.min()
    last_window = window_ids.max()

    update_points = np.arange(
        first_window + lookback_windows,
        last_window + 1,
        DELTA,
    )
    target_count = round(target_rate * lookback_windows / WINDOWS_PER_DAY)

    alert_mask = np.zeros(len(scores), dtype=bool)

    for index, update_window in enumerate(update_points):
        history_start = np.searchsorted(
            window_ids,
            update_window - lookback_windows,
            side="left",
        )
        history_end = np.searchsorted(window_ids, update_window, side="left")
        history_p = p_values[history_start:history_end]
        threshold = np.partition(history_p, target_count)[target_count]

        next_update = (
            update_points[index + 1]
            if index + 1 < len(update_points)
            else last_window + 1
        )
        apply_start = np.searchsorted(window_ids, update_window, side="left")
        apply_end = np.searchsorted(window_ids, next_update, side="left")
        alert_mask[apply_start:apply_end] = (
            p_values[apply_start:apply_end] < threshold
        )

    return alert_mask


scores = pd.read_parquet(
    SCORES_FILE,
    columns=[
        "day",
        "window_id",
        "account_type",
        "p_recal",
        "redteam_user_window_flag",
    ],
)

results = []

for lookback_hours in LOOKBACK_HOURS_LIST:
    lookback_windows = lookback_hours * WINDOWS_PER_HOUR
    for target_rate in TARGET_RATES:
        for account_type in ["human", "machine"]:
            account_scores = (
                scores[scores["account_type"] == account_type]
                .sort_values("window_id")
                .reset_index(drop=True)
            )

            alert_mask = controlled_rate(
                account_scores,
                target_rate,
                lookback_windows,
            )
            alerts = account_scores[alert_mask]

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

            results.append({
                "lookback_h": lookback_hours,
                "account_type": account_type,
                "r_target": target_rate,
                **weak_label,
                "eval_daily_mean": mean_daily,
                "eval_daily_sd": sd_daily,
                "eval_daily_cv": sd_daily / mean_daily,
            })


summary = pd.DataFrame(results)
summary.to_csv(OUTPUT_FILE, index=False)
print("\nControlled-rate summary (24h vs 168h lookback):")
print(summary.to_string(index=False), "\n")

print(f"saved {OUTPUT_FILE}")
