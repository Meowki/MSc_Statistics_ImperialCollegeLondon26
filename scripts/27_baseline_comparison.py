import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import *


MODEL_FILES = {
    "04 all-window": Path(
        "outputs/tables/04_human_poisson_scores.parquet"
    ),
    "25 back-solved": Path(
        "outputs/tables/25_human_backsolved_scores.parquet"
    ),
}
MODEL_KEYS = {
    "04 all-window": "04",
    "25 back-solved": "25",
}
MODEL_DISPLAY = {
    "04 all-window": "All-window baseline",
    "25 back-solved": "Activity-conditioned baseline",
}

OUT_SUMMARY = Path(
    "outputs/tables/27_baseline_summary.csv"
)
OUT_CENTERING = Path(
    "outputs/tables/27_centering_bins.csv"
)
FIG_CENTERING_NAME = "baseline_mean_alignment"

EVAL_DAYS = (44, 57)

ALPHAS = [0.05, 0.01, 0.005, 0.001]

P_FLOOR = 1e-300

OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)


def zt_poisson_midp(x, lam):
    p_nonzero = -np.expm1(-lam)
    p_mid = (
        poisson.sf(x - 1, lam)
        - 0.5 * poisson.pmf(x, lam)
    ) / p_nonzero
    return np.clip(p_mid, P_FLOOR, 1.0)


def zt_active_mean(lam):
    return lam / (-np.expm1(-lam))


def load_model(name, path):
    if name == "04 all-window":
        df = pd.read_parquet(
            path,
            columns=[
                "day",
                "event_count",
                "lambda_hat",
                "redteam_user_window_flag",
            ],
        )

        df["p_mid"] = zt_poisson_midp(
            df["event_count"].to_numpy(),
            df["lambda_hat"].to_numpy(),
        )

    else:
        df = pd.read_parquet(
            path,
            columns=[
                "day",
                "event_count",
                "lambda_hat",
                "p_mid",
                "redteam_user_window_flag",
            ],
        )

    # expected active-window count
    df["expected_active_mean"] = zt_active_mean(
        df["lambda_hat"].to_numpy()
    )
    return df


def make_summary(name, df):
    eval_df = df[
        (df["day"] >= EVAL_DAYS[0])
        & (df["day"] <= EVAL_DAYS[1])
    ]

    count = eval_df["event_count"].to_numpy()
    expected = eval_df["expected_active_mean"].to_numpy()
    p_mid = eval_df["p_mid"].to_numpy()

    row = {
        "model": name,
        "total_windows": len(df),
        "eval_windows": len(eval_df),
        "eligible_labelled_windows": int(
            df["redteam_user_window_flag"].sum()
        ),
        "median_count": float(np.median(count)),
        "median_lambda_hat": float(
            np.median(eval_df["lambda_hat"])
        ),
        "median_expected_active_mean": float(
            np.median(expected)
        ),
        "sum_count_over_expected": float(
            count.sum() / expected.sum()
        ),
        "mean_count_minus_expected": float(
            (count - expected).mean()
        ),
        "median_p_mid": float(np.median(p_mid)),
        "clipped": int((p_mid == P_FLOOR).sum()),
    }

    for alpha in ALPHAS:
        row[f"ter_{alpha}"] = (
            (p_mid < alpha).mean() / alpha
        )

    return row


def make_centering_bins(name, df):
    eval_df = df[
        (df["day"] >= EVAL_DAYS[0])
        & (df["day"] <= EVAL_DAYS[1])
    ][["event_count", "expected_active_mean"]].copy()

    eval_df["bin"] = pd.qcut(
        eval_df["expected_active_mean"],
        10,
        labels=False,
    )

    grouped = (
        eval_df.groupby("bin", as_index=False)
        .agg(
            windows=("event_count", "size"),
            observed_mean=("event_count", "mean"),
            expected_mean=(
                "expected_active_mean",
                "mean",
            ),
        )
    )
    grouped["model"] = name

    return grouped


summary_rows = []
centering_parts = []

for name, path in MODEL_FILES.items():
    print(f"loading {name}")
    df = load_model(name, path)

    summary_rows.append(
        make_summary(name, df)
    )
    centering_parts.append(
        make_centering_bins(name, df)
    )
    del df


summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_SUMMARY, index=False)

centering = pd.concat(
    centering_parts,
    ignore_index=True,
)
centering.to_csv(OUT_CENTERING, index=False)

model_names = list(MODEL_FILES)


print("\nBaseline summary, eval days 44-57:")
print(
    summary.round(4).to_string(index=False)
)

# observed and expected counts
fig, ax = plt.subplots(figsize=(HALF_W, 2.8))

for name in model_names:
    sub = centering[
        centering["model"] == name
    ].sort_values("expected_mean")

    color, marker, linestyle = MODEL_STYLE[MODEL_KEYS[name]]
    ax.plot(
        sub["expected_mean"],
        sub["observed_mean"],
        color=color,
        marker=marker,
        linestyle=linestyle,
        label=MODEL_DISPLAY[name],
    )

limit = max(
    centering["expected_mean"].max(),
    centering["observed_mean"].max(),
)

ax.plot(
    [0, limit],
    [0, limit],
    **REFERENCE_STYLE,
)
ax.set_xlabel("Expected active-window mean")
ax.set_ylabel("Observed active-window mean")
ax.legend(fontsize=7)
light_grid(ax)

save_fig(fig, FIG_CENTERING_NAME)
plt.close(fig)


print(f"\nsaved {OUT_SUMMARY}")
print(f"saved {OUT_CENTERING}")
print(f"saved {FIG_DIR / (FIG_CENTERING_NAME + '.pdf')}")
