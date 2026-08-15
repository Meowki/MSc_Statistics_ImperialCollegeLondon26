from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


WINDOWS_FILE = Path("outputs/tables/user_window_counts.parquet")
REDTEAM_FILE = Path("outputs/tables/redteam_user_window_labels.parquet")

HUMAN_OUT = Path("outputs/tables/25_human_backsolved_scores.parquet")
MACHINE_OUT = Path("outputs/tables/25_machine_backsolved_scores.parquet")

WINDOW_SEC = 300
HISTORY_DAYS = 7
WINDOWS_PER_DAY = 86400 // WINDOW_SEC
HISTORY_WINDOWS = HISTORY_DAYS * WINDOWS_PER_DAY

SEASONAL_FIT_DAYS = (30, 43)
EVAL_DAYS = (44, 57)

LAMBDA_FLOOR = 0.01
M_FLOOR = LAMBDA_FLOOR / (-np.expm1(-LAMBDA_FLOOR))
P_FLOOR = 1e-300
CHUNK_SIZE = 5_000_000
ALPHAS = [0.05, 0.01, 0.005, 0.001]

HUMAN_OUT.parent.mkdir(parents=True, exist_ok=True)


def add_time_features(df):
    window_start = df["window_id"] * WINDOW_SEC
    df["day"] = window_start // 86400
    df["hour"] = (window_start % 86400) // 3600
    df["dow"] = df["day"] % 7
    return df


def compute_active_seasonal_factors(df):
    # seasonal factors above the active-count minimum
    fitting_data = df[
        (df["day"] >= SEASONAL_FIT_DAYS[0])
        & (df["day"] <= SEASONAL_FIT_DAYS[1])
    ]

    cell_excess = (
        fitting_data.groupby(["hour", "dow"])["event_count"].mean() - 1.0
    )
    overall_excess = fitting_data["event_count"].mean() - 1.0

    return cell_excess / overall_excess


def add_active_mean(df):
    # seven-day active-window mean
    df = df.sort_values(["src_user", "window_id"]).copy()
    user_parts = []

    for _, user_data in df.groupby("src_user", sort=False):
        windows = user_data["window_id"].to_numpy()
        counts = user_data["event_count"].to_numpy()

        cumulative_counts = np.r_[0, np.cumsum(counts)]

        start = np.searchsorted(
            windows,
            windows - HISTORY_WINDOWS,
            side="left",
        )
        end = np.arange(len(windows))

        history_sum = cumulative_counts[end] - cumulative_counts[start]
        history_active_windows = (end - start).astype(float)
        history_active_windows[history_active_windows == 0] = np.nan

        user_data = user_data.copy()
        user_data["history_active_windows"] = history_active_windows
        user_data["m_raw"] = history_sum / history_active_windows
        user_parts.append(user_data)

    return pd.concat(user_parts, ignore_index=True)


def backsolve_lambda(m):
    # solve the zero-truncated mean equation
    lam = np.where(m < 1.5, 2.0 * (m - 1.0), m).astype(float)

    for _ in range(6):
        exp_neg = np.exp(-lam)
        denominator = -np.expm1(-lam)

        value = lam / denominator - m
        derivative = (
            denominator - lam * exp_neg
        ) / denominator**2

        lam = lam - value / derivative

    return lam


def backsolve_in_chunks(m):
    result = np.empty_like(m, dtype=float)

    for start in range(0, len(m), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(m))
        result[start:end] = backsolve_lambda(m[start:end])

    return result


def zt_poisson_midp(x, lam):
    p_nonzero = -np.expm1(-lam)

    p_mid = (
        poisson.sf(x - 1, lam)
        - 0.5 * poisson.pmf(x, lam)
    ) / p_nonzero

    return np.clip(p_mid, P_FLOOR, 1.0)


def score_accounts(df, seasonal):
    df = add_active_mean(df)
    df = df.dropna(subset=["m_raw"])

    df["seasonal_factor"] = (
        df.set_index(["hour", "dow"])
        .index.map(seasonal)
        .values
    )

    # active-mean seasonal adjustment
    df["m_hat"] = (
        1.0
        + (df["m_raw"] - 1.0) * df["seasonal_factor"]
    ).clip(lower=M_FLOOR)

    df["lambda_hat"] = backsolve_in_chunks(
        df["m_hat"].to_numpy()
    )

    df["p_mid"] = zt_poisson_midp(
        df["event_count"].to_numpy(),
        df["lambda_hat"].to_numpy(),
    )
    df["score"] = -np.log10(df["p_mid"])

    return df


def print_diagnostic(df, account_type):
    eval_df = df[
        (df["day"] >= EVAL_DAYS[0])
        & (df["day"] <= EVAL_DAYS[1])
    ]

    count = eval_df["event_count"].to_numpy()
    m_hat = eval_df["m_hat"].to_numpy()
    p_mid = eval_df["p_mid"].to_numpy()

    print(f"\n{account_type}, eval days {EVAL_DAYS[0]}-{EVAL_DAYS[1]}")
    print(f"windows: {len(eval_df):,}")
    print(f"median count: {np.median(count):.3f}")
    print(f"median m_hat: {np.median(m_hat):.3f}")
    print(f"median lambda_hat: {np.median(eval_df['lambda_hat']):.3f}")
    print(f"sum(count) / sum(m_hat): {count.sum() / m_hat.sum():.3f}")
    print(f"mean(count - m_hat): {(count - m_hat).mean():.3f}")
    print(f"clipped: {(p_mid == P_FLOOR).sum():,}")

    for alpha in ALPHAS:
        ter = (p_mid < alpha).mean() / alpha
        print(f"alpha={alpha:.3f}  TER={ter:.2f}")


columns = [
    "src_user",
    "window_id",
    "account_type",
    "event_count",
    "failure_count",
]

# human seasonal factors from the fitting period
human = pd.read_parquet(
    WINDOWS_FILE,
    columns=columns,
    filters=[("account_type", "==", "human")],
)
human = add_time_features(human)

seasonal = compute_active_seasonal_factors(human)

human = score_accounts(human, seasonal)

redteam = pd.read_parquet(REDTEAM_FILE)[
    ["src_user", "window_id", "redteam_user_window_flag"]
]
human = human.merge(
    redteam,
    on=["src_user", "window_id"],
    how="left",
)
human["redteam_user_window_flag"] = (
    human["redteam_user_window_flag"]
    .fillna(0)
    .astype(np.int8)
)

keep = [
    "src_user",
    "window_id",
    "day",
    "hour",
    "dow",
    "account_type",
    "event_count",
    "failure_count",
    "history_active_windows",
    "m_raw",
    "seasonal_factor",
    "m_hat",
    "lambda_hat",
    "p_mid",
    "score",
    "redteam_user_window_flag",
]

human[keep].to_parquet(HUMAN_OUT, index=False)

print(f"human scored windows: {len(human):,}")
print_diagnostic(human, "human")

del human


# machine scores with human seasonal factors
machine = pd.read_parquet(
    WINDOWS_FILE,
    columns=columns,
    filters=[("account_type", "==", "machine")],
)
machine = add_time_features(machine)
machine = score_accounts(machine, seasonal)
machine["redteam_user_window_flag"] = np.int8(0)

machine[keep].to_parquet(MACHINE_OUT, index=False)

print(f"\nmachine scored windows: {len(machine):,}")
print_diagnostic(machine, "machine")

print(f"\nsaved {HUMAN_OUT}")
print(f"saved {MACHINE_OUT}")
