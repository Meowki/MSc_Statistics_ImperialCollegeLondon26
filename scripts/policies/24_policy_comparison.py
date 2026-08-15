import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotstyle import *

OPERATING_POINTS_FILE = Path("outputs/tables/24_policy_operating_points.csv")
MATCHED_BURDEN_FILE = Path("outputs/tables/24_matched_burden_cv.csv")
FIG_OVERLAP_NAME = "policy_operating_overlap_human"
FIG_CV_NAMES = {
    "human": "policy_operating_cv_human",
    "machine": "policy_operating_cv_machine",
}
FIG_MATCHED_NAMES = {
    ("human", "low"): "matched_burden_cv_human_low",
    ("human", "high"): "matched_burden_cv_human_high",
    ("machine", "low"): "matched_burden_cv_machine_low",
    ("machine", "high"): "matched_burden_cv_machine_high",
}

LABELLED_DAYS_COUNT = 18
EVAL_DAYS = (44, 57)
TARGET_BURDENS = {
    "human": {"low": 300, "high": 3000},
    "machine": {"low": 500, "high": 8000},
}

OPERATING_POINTS_FILE.parent.mkdir(parents=True, exist_ok=True)


def overlap_values(account_type, days_hit):
    if account_type != "human" or pd.isna(days_hit):
        return np.nan, np.nan
    days_hit = int(days_hit)
    return days_hit, days_hit / LABELLED_DAYS_COUNT


# policy operating points
points = []

# fixed thresholds
fixed_results = pd.read_csv("outputs/tables/23_policy_stratified_summary.csv")
for _, row in fixed_results.iterrows():
    days_hit, overlap = overlap_values(
        row["account_type"], row["labelled_days_hit"]
    )
    points.append({
        "policy": f"fixed_{row['variant']}",
        "param": float(row["alpha"]),
        "account_type": row["account_type"],
        "burden": float(row["eval_daily_mean"]),
        "cv": float(row["eval_daily_cv"]),
        "labelled_days_hit": days_hit,
        "labelled_day_overlap": overlap,
    })

# per-type Top-K
topk_results = pd.read_csv("outputs/tables/20_policy_topk_summary.csv")
for _, row in topk_results[topk_results["variant"] == "per_type"].iterrows():
    per_type_k = int(row["K"]) // 2
    for account_type in ["human", "machine"]:
        days_hit, overlap = overlap_values(
            account_type,
            row["covered_days"] if account_type == "human" else np.nan,
        )
        points.append({
            "policy": "topk_per_type",
            "param": int(row["K"]),
            "account_type": account_type,
            "burden": float(per_type_k),
            "cv": 0.0,
            "labelled_days_hit": days_hit,
            "labelled_day_overlap": overlap,
        })

# pooled Top-K
composition = pd.read_csv("outputs/tables/20_policy_topk_composition.csv")
evaluation_composition = composition[
    (composition["day"] >= EVAL_DAYS[0])
    & (composition["day"] <= EVAL_DAYS[1])
]
for _, row in topk_results[topk_results["variant"] == "global"].iterrows():
    k = int(row["K"])
    daily_composition = evaluation_composition[
        (evaluation_composition["variant"] == "global")
        & (evaluation_composition["K"] == k)
    ]
    for account_type in ["human", "machine"]:
        daily_alerts = (
            daily_composition.set_index("day")[account_type]
            .reindex(range(EVAL_DAYS[0], EVAL_DAYS[1] + 1), fill_value=0)
        )
        mean_daily = float(daily_alerts.mean())
        sd_daily = float(daily_alerts.std())
        days_hit, overlap = overlap_values(
            account_type,
            row["covered_days"] if account_type == "human" else np.nan,
        )
        points.append({
            "policy": "topk_global",
            "param": k,
            "account_type": account_type,
            "burden": mean_daily,
            "cv": sd_daily / mean_daily if mean_daily > 0 else np.nan,
            "labelled_days_hit": days_hit,
            "labelled_day_overlap": overlap,
        })

# controlled-rate threshold
rate_results = pd.read_csv("outputs/tables/21_policy_rate_summary.csv")
rate_results = rate_results[rate_results["lookback_h"] == 168]
for _, row in rate_results.iterrows():
    days_hit, overlap = overlap_values(
        row["account_type"], row["labelled_days_hit"]
    )
    points.append({
        "policy": "controlled_rate_168h",
        "param": int(row["r_target"]),
        "account_type": row["account_type"],
        "burden": float(row["eval_daily_mean"]),
        "cv": float(row["eval_daily_cv"]),
        "labelled_days_hit": days_hit,
        "labelled_day_overlap": overlap,
    })

# exploratory Q-chart
qchart_results = pd.read_csv("outputs/tables/22_policy_qchart_summary.csv")
qchart_results = qchart_results[
    (qchart_results["gamma"] == 0.25)
    & (qchart_results["eval_daily_mean"] > 0)
]
for _, row in qchart_results.iterrows():
    days_hit, overlap = overlap_values(
        row["account_type"], row["labelled_days_hit"]
    )
    points.append({
        "policy": "qchart_ewma",
        "param": float(row["tau"]),
        "account_type": row["account_type"],
        "burden": float(row["eval_daily_mean"]),
        "cv": float(row["eval_daily_cv"]),
        "labelled_days_hit": days_hit,
        "labelled_day_overlap": overlap,
    })


operating_points = pd.DataFrame(points)
operating_points = operating_points[operating_points["burden"] > 0]
operating_points.to_csv(OPERATING_POINTS_FILE, index=False)
print(f"total operating points: {len(operating_points)}")


POLICY_DISPLAY = {
    "fixed_global": "Fixed (global)",
    "fixed_stratified": "Fixed (stratified)",
    "topk_global": "Top-K (global)",
    "topk_per_type": "Top-K (per type)",
    "controlled_rate_168h": "Controlled-rate (168h)",
    "qchart_ewma": "Q-chart EWMA (exploratory)",
}

POLICY_TICK = {
    "fixed_global": "Fixed\n(global)",
    "fixed_stratified": "Fixed\n(stratified)",
    "topk_global": "Top-K\n(global)",
    "topk_per_type": "Top-K\n(per type)",
    "controlled_rate_168h": "Controlled\nrate",
    "qchart_ewma": "Q-chart\nEWMA",
}

# distinct controlled-rate colour
SEAGLASS = "#379F9F"
POLICY_STYLE = dict(POLICY_STYLE, controlled_rate_168h=(SEAGLASS, "v", "-."))
POLICY_COLOR = dict(POLICY_COLOR, controlled_rate_168h=SEAGLASS)


def interp_cv_at_burden(sub, target_burden):
    sub = sub.groupby("burden", as_index=False)["cv"].mean().sort_values("burden")
    burdens = sub["burden"].to_numpy()
    cv_values = sub["cv"].to_numpy()
    if target_burden < burdens.min() or target_burden > burdens.max():
        return np.nan
    log_burdens = np.log10(burdens)
    return float(
        np.interp(np.log10(target_burden), log_burdens, cv_values)
    )


def plot_policy_curve(ax, sub, y_col, policy):
    color, marker, linestyle = POLICY_STYLE[policy]
    alpha = 0.65 if policy == "qchart_ewma" else 1.0
    ax.plot(
        sub["burden"],
        sub[y_col],
        color=color,
        marker=marker,
        linestyle=linestyle,
        alpha=alpha,
        label=POLICY_DISPLAY[policy],
    )


# operating curves
fig, ax = plt.subplots(figsize=(FULL_W, 2.5))
for policy in POLICY_DISPLAY:
    sub = operating_points[
        (operating_points["policy"] == policy)
        & (operating_points["account_type"] == "human")
    ].sort_values("burden")
    if len(sub):
        plot_policy_curve(ax, sub, "labelled_day_overlap", policy)

ax.set_xscale("log")
ax.set_xlabel("Mean daily burden (alerts/day)")
ax.set_ylabel("Labelled-day overlap")
ax.set_ylim(0, 0.78)
light_grid(ax)
ax.set_axisbelow(True)
ax.legend(loc="upper left", ncol=2, fontsize=7, columnspacing=1.0, handlelength=2.2)
save_fig(fig, FIG_OVERLAP_NAME)
plt.close(fig)

for account_type in ["human", "machine"]:
    fig, ax = plt.subplots(figsize=(HALF_W, 2.5))
    for policy in POLICY_DISPLAY:
        sub = operating_points[
            (operating_points["policy"] == policy)
            & (operating_points["account_type"] == account_type)
        ].sort_values("burden")
        if len(sub):
            plot_policy_curve(ax, sub, "cv", policy)
    ax.set_xscale("log")
    ax.set_xlabel("Mean daily burden (alerts/day)")
    ax.set_ylabel("Daily CV")
    ax.set_ylim(bottom=0)
    light_grid(ax)
    ax.set_axisbelow(True)
    save_fig(fig, FIG_CV_NAMES[account_type])
    plt.close(fig)


# matched-burden CV
matched_rows = []
for account_type in ["human", "machine"]:
    for region, target in TARGET_BURDENS[account_type].items():
        for policy in POLICY_DISPLAY:
            sub = operating_points[
                (operating_points["policy"] == policy)
                & (operating_points["account_type"] == account_type)
            ]
            if len(sub) == 0:
                continue
            interpolated_cv = interp_cv_at_burden(sub, target)
            if np.isnan(interpolated_cv):
                continue
            matched_rows.append({
                "account_type": account_type,
                "region": region,
                "target_burden": target,
                "policy": policy,
                "cv_interp": round(interpolated_cv, 3),
            })

matched = pd.DataFrame(matched_rows)
matched.to_csv(MATCHED_BURDEN_FILE, index=False)
print("Matched-burden CV (two-region, log-burden interpolation):")
print(matched.to_string(index=False), "\n")


# matched-burden panels
POLICY_ORDER = list(POLICY_DISPLAY)

for account_type in ["human", "machine"]:
    for region in ["low", "high"]:
        fig, ax = plt.subplots(figsize=(HALF_W, 2.5))
        sub = matched[
            (matched["account_type"] == account_type)
            & (matched["region"] == region)
        ]
        # consistent policy order
        sub = sub.set_index("policy").reindex(POLICY_ORDER).dropna(subset=["cv_interp"])
        x = np.arange(len(sub))
        bars = ax.bar(
            x,
            sub["cv_interp"].values,
            width=0.62,
            color=[POLICY_COLOR[p] for p in sub.index],
        )
        for bar, policy in zip(bars, sub.index):
            if policy == "qchart_ewma":
                bar.set_alpha(0.55)
        ymax = max(sub["cv_interp"].max() * 1.30, 0.05) if len(sub) else 0.05
        # zero-value labels
        for xi, v in zip(x, sub["cv_interp"].values):
            ax.text(xi, v + 0.02 * ymax, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_TICK[p] for p in sub.index], fontsize=6.8)
        ax.set_ylabel("Daily CV")
        ax.set_ylim(0, ymax)
        light_grid(ax, axis="y")
        ax.set_axisbelow(True)
        save_fig(fig, FIG_MATCHED_NAMES[(account_type, region)])
        plt.close(fig)


print(f"saved {OPERATING_POINTS_FILE}")
print(f"saved {MATCHED_BURDEN_FILE}")
print(f"saved {FIG_DIR / (FIG_OVERLAP_NAME + '.pdf')}")
for name in FIG_CV_NAMES.values():
    print(f"saved {FIG_DIR / (name + '.pdf')}")
for name in FIG_MATCHED_NAMES.values():
    print(f"saved {FIG_DIR / (name + '.pdf')}")
