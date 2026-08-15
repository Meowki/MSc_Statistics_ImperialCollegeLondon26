from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

WINDOWS_FILE = Path("outputs/tables/user_window_counts.parquet")
REDTEAM_FILE = Path("outputs/tables/redteam_user_window_labels.parquet")
OUTPUT_FILE = Path("outputs/tables/04_human_poisson_scores.parquet")

WINDOW_SECONDS = 300
HISTORY_DAYS = 7
WINDOWS_PER_DAY = 86400 // WINDOW_SECONDS
HISTORY_WINDOWS = HISTORY_DAYS * WINDOWS_PER_DAY
LAMBDA_FLOOR = 0.01
P_FLOOR = 1e-300

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def add_time_features(data):
    window_start = data["window_id"] * WINDOW_SECONDS
    data["day"] = window_start // 86400
    data["hour"] = (window_start % 86400) // 3600
    data["dow"] = data["day"] % 7
    return data


def compute_seasonal_factors(data):
    # total human traffic by window
    window_totals = (
        data.groupby(["window_id", "hour", "dow"], as_index=False)["event_count"]
        .sum()
    )

    overall_mean = window_totals["event_count"].mean()

    # hour-of-week traffic
    cell_means = (
        window_totals
        .groupby(["hour", "dow"])["event_count"]
        .mean()
    )

    return cell_means / overall_mean


def add_rolling_rate(data):
    data = data.sort_values(["src_user", "window_id"]).copy()
    user_parts = []

    for _, user_data in data.groupby("src_user", sort=False):
        windows = user_data["window_id"].to_numpy()
        counts = user_data["event_count"].to_numpy()

        # rolling count sums
        cumulative_counts = np.r_[0, np.cumsum(counts)]

        # seven-day history
        start = np.searchsorted(windows, windows - HISTORY_WINDOWS, side="left")
        end = np.arange(len(windows))
        history_sum = cumulative_counts[end] - cumulative_counts[start]

        history_windows = np.minimum(windows, HISTORY_WINDOWS).astype(float)
        history_windows[history_windows == 0] = np.nan

        user_data = user_data.copy()
        user_data["lambda_raw"] = history_sum / history_windows
        user_data["history_windows"] = history_windows
        user_parts.append(user_data)

    return pd.concat(user_parts, ignore_index=True)


def zt_poisson_pvalue(count, rate):
    # conditional upper-tail probability
    p_upper = poisson.sf(count - 1, rate)
    p_nonzero = -np.expm1(-rate)
    return np.clip(p_upper / p_nonzero, P_FLOOR, 1.0)


scores = pd.read_parquet(WINDOWS_FILE)
scores = scores[scores["account_type"] == "human"].copy()
scores = add_time_features(scores)

seasonal_factors = compute_seasonal_factors(scores)

scores = add_rolling_rate(scores)

# hour-of-week adjustment
scores["seasonal_factor"] = (
    scores.set_index(["hour", "dow"]).index.map(seasonal_factors).values
)

scores["lambda_hat"] = (
    scores["lambda_raw"] * scores["seasonal_factor"]
).clip(lower=LAMBDA_FLOOR)

scores = scores.dropna(subset=["lambda_raw"])

p = zt_poisson_pvalue(
    scores["event_count"].to_numpy(),
    scores["lambda_hat"].to_numpy(),
)
scores["p_value"] = p
scores["score"] = -np.log10(p)

# red-team labels
redteam_labels = pd.read_parquet(REDTEAM_FILE)[
    ["src_user", "window_id", "redteam_user_window_flag"]
]
scores = scores.merge(redteam_labels, on=["src_user", "window_id"], how="left")
scores["redteam_user_window_flag"] = (
    scores["redteam_user_window_flag"].fillna(0).astype(np.int8)
)

keep = [
    "src_user", "window_id", "day", "hour", "dow",
    "account_type", "event_count", "failure_count",
    "history_windows", "lambda_raw", "seasonal_factor",
    "lambda_hat", "p_value", "score",
    "redteam_user_window_flag",
]
scores[keep].to_parquet(OUTPUT_FILE, index=False)

print(f"human scored windows: {len(scores):,}")
print(f"red-team windows: {scores['redteam_user_window_flag'].sum()}")
print()
print(scores[["event_count", "lambda_hat", "p_value", "score"]].describe().round(3))
