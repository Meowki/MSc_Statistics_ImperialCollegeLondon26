import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_utils import add_tie_key

SCORES_FILE = Path("outputs/tables/26_recalibrated_all.parquet")
SUMMARY_FILE = Path("outputs/tables/20_policy_topk_summary.csv")
COMPOSITION_FILE = Path("outputs/tables/20_policy_topk_composition.csv")

K_VALUES = [10, 50, 100, 500, 1000]
EVAL_DAYS = (44, 57)
LABELLED_DAYS = (0, 29)

SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)


def evaluate(alerts, k, variant, labelled_days_total):
    labelled_mask = (
        (alerts["day"] >= LABELLED_DAYS[0])
        & (alerts["day"] <= LABELLED_DAYS[1])
    )
    eval_mask = (alerts["day"] >= EVAL_DAYS[0]) & (alerts["day"] <= EVAL_DAYS[1])
    labelled_hits = alerts[
        labelled_mask & (alerts["redteam_user_window_flag"] == 1)
    ]

    return {
        "variant": variant,
        "K": k,
        "covered_days": labelled_hits["day"].nunique(),
        "labeled_days": labelled_days_total,
        "rt_hits_labeled": int(labelled_hits["redteam_user_window_flag"].sum()),
        "total_alerts_labeled": int(labelled_mask.sum()),
        "alerts_per_day_eval": int(eval_mask.sum()) / 14,
    }


# recalibrated candidates
scores = pd.read_parquet(
    SCORES_FILE,
    columns=[
        "src_user",
        "window_id",
        "day",
        "account_type",
        "p_recal",
        "redteam_user_window_flag",
    ],
)
labelled_days_total = scores.loc[
    scores["redteam_user_window_flag"] == 1,
    "day",
].nunique()

# shared tie-breaking order
scores = add_tie_key(scores).drop(columns=["src_user", "window_id"])
scores = scores.sort_values(
    ["day", "p_recal", "_tie_key"],
    kind="mergesort",
).reset_index(drop=True)
human_scores = scores[scores["account_type"] == "human"]
machine_scores = scores[scores["account_type"] == "machine"]


results = []
comp_rows = []

for k in K_VALUES:
    # pooled daily Top-K
    pooled_alerts = scores.groupby("day").head(k)
    results.append(evaluate(pooled_alerts, k, "global", labelled_days_total))

    composition = (
        pooled_alerts
        .groupby(["day", "account_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["human", "machine"], fill_value=0)
    )
    for day, row in composition.iterrows():
        comp_rows.append({
            "variant": "global",
            "K": k,
            "day": int(day),
            "human": int(row["human"]),
            "machine": int(row["machine"]),
        })

    # equal account-type allocation
    per_type_k = k // 2
    human_alerts = human_scores.groupby("day").head(per_type_k)
    machine_alerts = machine_scores.groupby("day").head(per_type_k)
    per_type_alerts = pd.concat([human_alerts, machine_alerts], ignore_index=True)
    results.append(evaluate(per_type_alerts, k, "per_type", labelled_days_total))


summary = pd.DataFrame(results)
summary.to_csv(SUMMARY_FILE, index=False)
print("\nTop-K summary:")
print(summary.round(5).to_string(index=False), "\n")

composition = pd.DataFrame(comp_rows)
composition.to_csv(COMPOSITION_FILE, index=False)

print(f"saved {SUMMARY_FILE}")
print(f"saved {COMPOSITION_FILE}")
