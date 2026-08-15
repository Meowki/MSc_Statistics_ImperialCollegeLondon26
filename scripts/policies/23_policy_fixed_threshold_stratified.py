import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_utils import labelled_metrics

SCORES_FILE = Path("outputs/tables/26_recalibrated_all.parquet")
OUTPUT_FILE = Path("outputs/tables/23_policy_stratified_summary.csv")

ALPHAS = [0.05, 0.01, 0.005, 0.001]
EVAL_DAYS = (44, 57)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


scores = pd.read_parquet(
    SCORES_FILE,
    columns=[
        "day",
        "account_type",
        "p_recal",
        "p_recal_bin",
        "redteam_user_window_flag",
    ],
)

results = []

for alpha in ALPHAS:
    for account_type in ["human", "machine"]:
        account_scores = scores[scores["account_type"] == account_type]
        for variant, p_column in [
            ("global", "p_recal"),
            ("stratified", "p_recal_bin"),
        ]:
            alerts = account_scores[account_scores[p_column] < alpha]
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
                "alpha": alpha,
                "account_type": account_type,
                "variant": variant,
                **weak_label,
                "eval_daily_mean": mean_daily,
                "eval_daily_sd": sd_daily,
                "eval_daily_cv": sd_daily / mean_daily,
            })


summary = pd.DataFrame(results)
summary.to_csv(OUTPUT_FILE, index=False)
print("Time-stratified vs global recalibration (eval span):")
print(summary[summary["alpha"].isin([0.01, 0.005, 0.001])]
      .sort_values(["account_type", "alpha", "variant"])
      .to_string(index=False), "\n")


print(f"saved {OUTPUT_FILE}")
