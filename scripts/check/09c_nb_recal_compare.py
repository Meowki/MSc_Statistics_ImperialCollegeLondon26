import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb_utils import match_nb_mean, zt_nb_midp
from policy_utils import add_tie_key
from recalibration_utils import empirical_rank_map


SCORES_FILE = Path("outputs/tables/26_recalibrated_all.parquet")
NB_RESULTS_FILE = Path("outputs/tables/09b_nb_matched_ter.csv")
TER_OUTPUT_FILE = Path("outputs/tables/09c_recal_ter_compare.csv")
POLICY_OUTPUT_FILE = Path("outputs/tables/09c_policy_impact.csv")

FIT_DAYS = (30, 43)
EVAL_DAYS = (44, 57)
LABELLED_DAYS = (0, 29)
ALPHAS = [0.05, 0.01, 0.005, 0.001]
FIXED_THRESHOLDS = [0.005, 0.001]
TOPK_PER_TYPE = 250
RANK_SAMPLE_SIZE = 200_000


def labelled_day_count(data, alert_mask):
    return data.loc[
        alert_mask & (data["redteam_user_window_flag"] == 1),
        "day",
    ].nunique()


scores = pd.read_parquet(
    SCORES_FILE,
    columns=[
        "src_user",
        "window_id",
        "day",
        "event_count",
        "m_hat",
        "p_recal",
        "redteam_user_window_flag",
    ],
    filters=[("account_type", "==", "human")],
)

fitted_size = pd.read_csv(NB_RESULTS_FILE)["r_hat"].iloc[0]

# negative binomial scores
nb_means = match_nb_mean(scores["m_hat"].to_numpy(), fitted_size)
scores["p_nb"] = zt_nb_midp(
    scores["event_count"].to_numpy(),
    nb_means,
    fitted_size,
)

# negative binomial rank map
fit_mask = (
    (scores["day"] >= FIT_DAYS[0])
    & (scores["day"] <= FIT_DAYS[1])
)
scores["p_nb_recal"] = empirical_rank_map(
    scores.loc[fit_mask, "p_nb"].to_numpy(),
    scores["p_nb"].to_numpy(),
)

# evaluation-period comparison
evaluation_data = scores[
    (scores["day"] >= EVAL_DAYS[0])
    & (scores["day"] <= EVAL_DAYS[1])
]

ter_rows = []
for alpha in ALPHAS:
    poisson_ter = (evaluation_data["p_recal"] < alpha).mean() / alpha
    nb_ter = (evaluation_data["p_nb_recal"] < alpha).mean() / alpha
    ter_rows.append({
        "alpha": alpha,
        "ter_poisson_recal": poisson_ter,
        "ter_nb_recal": nb_ter,
        "r_hat": fitted_size,
        "eval_windows": len(evaluation_data),
    })

rank_sample = evaluation_data.sample(RANK_SAMPLE_SIZE, random_state=1)
rank_correlation = spearmanr(
    rank_sample["p_recal"],
    rank_sample["p_nb_recal"],
).correlation

ter_results = pd.DataFrame(ter_rows)
ter_results["spearman_recal"] = rank_correlation
ter_results.to_csv(TER_OUTPUT_FILE, index=False)

# labelled-period policy comparison
labelled_data = scores[
    (scores["day"] >= LABELLED_DAYS[0])
    & (scores["day"] <= LABELLED_DAYS[1])
].copy()
labelled_days_total = labelled_data.loc[
    labelled_data["redteam_user_window_flag"] == 1,
    "day",
].nunique()

policy_rows = []
for alpha in FIXED_THRESHOLDS:
    poisson_alerts = labelled_data["p_recal"] < alpha
    nb_alerts = labelled_data["p_nb_recal"] < alpha
    intersection = (poisson_alerts & nb_alerts).sum()
    union = (poisson_alerts | nb_alerts).sum()

    policy_rows.append({
        "policy": "fixed_threshold",
        "parameter": alpha,
        "poisson_covered_days": labelled_day_count(labelled_data, poisson_alerts),
        "nb_covered_days": labelled_day_count(labelled_data, nb_alerts),
        "labelled_days_total": labelled_days_total,
        "poisson_alerts": int(poisson_alerts.sum()),
        "nb_alerts": int(nb_alerts.sum()),
        "selection_overlap": intersection / union,
    })

# per-type Top-K comparison
labelled_data = add_tie_key(labelled_data)
poisson_topk = (
    labelled_data
    .sort_values(["day", "p_recal", "_tie_key"])
    .groupby("day")
    .head(TOPK_PER_TYPE)
)
nb_topk = (
    labelled_data
    .sort_values(["day", "p_nb_recal", "_tie_key"])
    .groupby("day")
    .head(TOPK_PER_TYPE)
)

poisson_index = set(poisson_topk.index)
nb_index = set(nb_topk.index)
policy_rows.append({
    "policy": "topk_per_type",
    "parameter": TOPK_PER_TYPE,
    "poisson_covered_days": (
        poisson_topk.loc[poisson_topk["redteam_user_window_flag"] == 1, "day"].nunique()
    ),
    "nb_covered_days": (
        nb_topk.loc[nb_topk["redteam_user_window_flag"] == 1, "day"].nunique()
    ),
    "labelled_days_total": labelled_days_total,
    "poisson_alerts": len(poisson_topk),
    "nb_alerts": len(nb_topk),
    "selection_overlap": (
        len(poisson_index & nb_index) / len(poisson_index | nb_index)
    ),
})

policy_results = pd.DataFrame(policy_rows)
policy_results.to_csv(POLICY_OUTPUT_FILE, index=False)

print(f"fitted NB size r: {fitted_size:.3f}")
print(f"recalibrated rank correlation: {rank_correlation:.4f}")
print("\nTER comparison")
print(ter_results.round(4).to_string(index=False))
print("\npolicy comparison")
print(policy_results.round(4).to_string(index=False))
print(f"\nsaved {TER_OUTPUT_FILE}")
print(f"saved {POLICY_OUTPUT_FILE}")
