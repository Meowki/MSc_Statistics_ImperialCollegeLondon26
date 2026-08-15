import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import *
from recalibration_utils import empirical_rank_map


HUMAN_FILE = Path("outputs/tables/25_human_backsolved_scores.parquet")
MACHINE_FILE = Path("outputs/tables/25_machine_backsolved_scores.parquet")

OUT_ALL = Path("outputs/tables/26_recalibrated_all.parquet")
OUT_TER = Path("outputs/tables/26_recalibration_ter.csv")
OUT_COND_TER = Path("outputs/tables/26_conditional_ter.csv")
FIG_MARGINAL_PREFIX = "recalibration_marginal"
FIG_CELLS_PREFIX = "recalibration_cells"

FIT_DAYS = (30, 43)
EVAL_DAYS = (44, 57)
ALPHAS = [0.05, 0.01, 0.005, 0.001]
REPRESENTATIVE_ALPHA = 0.01
P_FLOOR_Z = 1e-10

OUT_ALL.parent.mkdir(parents=True, exist_ok=True)


def recalibrate_stratified(data):
    # hour-of-week rank maps
    recalibrated = np.empty(len(data))

    for _, cell_data in data.groupby(["hour", "dow"]):
        fit_mask = (
            (cell_data["day"] >= FIT_DAYS[0])
            & (cell_data["day"] <= FIT_DAYS[1])
        )
        fit_p = cell_data.loc[fit_mask, "p_mid"].to_numpy()
        recalibrated[cell_data.index] = empirical_rank_map(
            fit_p,
            cell_data["p_mid"].to_numpy(),
        )

    return recalibrated


def compute_ter(p):
    rows = []
    for a in ALPHAS:
        observed_rate = float((p < a).mean())
        rows.append({
            "alpha": a,
            "observed_rate": observed_rate,
            "ter": observed_rate / a,
        })
    return rows


def process_scores(account_type, path):
    df = pd.read_parquet(path)
    df = df.reset_index(drop=True)

    fit_mask = (df["day"] >= FIT_DAYS[0]) & (df["day"] <= FIT_DAYS[1])
    eval_mask = (df["day"] >= EVAL_DAYS[0]) & (df["day"] <= EVAL_DAYS[1])

    # account-type map
    fit_p = df.loc[fit_mask, "p_mid"].to_numpy()
    df["p_recal"] = empirical_rank_map(fit_p, df["p_mid"].to_numpy())

    # time-stratified map
    df["p_recal_bin"] = recalibrate_stratified(df)

    # normal scores for Q-chart
    p_clipped = np.clip(df["p_recal"].to_numpy(), P_FLOOR_Z, 1 - P_FLOOR_Z)
    df["z"] = norm.isf(p_clipped)

    df["account_type"] = account_type

    # marginal TER
    raw_eval = df.loc[eval_mask, "p_mid"].to_numpy()
    recal_eval = df.loc[eval_mask, "p_recal"].to_numpy()
    strat_eval = df.loc[eval_mask, "p_recal_bin"].to_numpy()

    ter_rows = []
    for stage, p in [("raw", raw_eval), ("global", recal_eval), ("stratified", strat_eval)]:
        for row in compute_ter(p):
            ter_rows.append({"account_type": account_type, "stage": stage, **row})

    # cell-level TER
    cond_rows = []
    eval_df = df[eval_mask]
    for (h, d), g in eval_df.groupby(["hour", "dow"]):
        for stage, col in [("global", "p_recal"), ("stratified", "p_recal_bin")]:
            p = g[col].to_numpy()
            for a in ALPHAS:
                cond_rows.append({
                    "account_type": account_type,
                    "hour": int(h), "dow": int(d),
                    "stage": stage, "alpha": a,
                    "n": len(p),
                    "ter": (p < a).mean() / a,
                })

    return df, ter_rows, cond_rows


human_df, human_ter, human_cond = process_scores("human", HUMAN_FILE)

machine_df, machine_ter, machine_cond = process_scores("machine", MACHINE_FILE)

# combined policy input
all_df = pd.concat([human_df, machine_df], ignore_index=True)
all_df.to_parquet(OUT_ALL, index=False)
print(f"\nmerged all: {len(all_df):,} rows")

# marginal TER table
ter_df = pd.DataFrame(human_ter + machine_ter)
ter_df.to_csv(OUT_TER, index=False)
print("\nTER on eval days 44-57:")
print(ter_df.round(4).to_string(index=False))

# cell-level TER table
cond_df = pd.DataFrame(human_cond + machine_cond)
cond_df.to_csv(OUT_COND_TER, index=False)

# cell-level TER summary
print("\nConditional TER spread across (hour,dow) bins, eval:")
for t in ["human", "machine"]:
    for a in [REPRESENTATIVE_ALPHA]:
        for stage in ["global", "stratified"]:
            sub = cond_df[
                (cond_df["account_type"] == t)
                & (cond_df["alpha"] == a)
                & (cond_df["stage"] == stage)
            ]
            ter_vals = sub["ter"].to_numpy()
            print(
                f"  {t} a={a} {stage:10s}: "
                f"mean={ter_vals.mean():.2f} sd={ter_vals.std():.2f} "
                f"min={ter_vals.min():.2f} max={ter_vals.max():.2f}"
            )


stage_style = {
    "raw": CAL_STYLE["raw"],
    "global": CAL_STYLE["recal_global"],
    "stratified": CAL_STYLE["recal_stratified"],
}
stage_label = {
    "raw": "Raw",
    "global": "Account-type map",
    "stratified": "Time-stratified map",
}
# marginal TER panels
x = np.arange(len(ALPHAS))
width = 0.24
for t in ["human", "machine"]:
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    for i, stage in enumerate(["raw", "global", "stratified"]):
        sub = (
            ter_df[(ter_df["account_type"] == t) & (ter_df["stage"] == stage)]
            .set_index("alpha")
            .loc[ALPHAS]
        )
        color, _, _ = stage_style[stage]
        ax.bar(
            x + (i - 1) * width,
            sub["ter"],
            width,
            color=color,
            label=stage_label[stage],
        )
    ax.axhline(1.0, **REFERENCE_STYLE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in ALPHAS])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Marginal TER")
    ax.set_yscale("log")
    light_grid(ax, axis="y")
    if t == "human":
        ax.legend(fontsize=6.5, ncol=1, loc="upper left")
    save_fig(fig, f"{FIG_MARGINAL_PREFIX}_{t}")
    plt.close(fig)

# cell-level TER panels
for t in ["human", "machine"]:
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    cell_ter = cond_df[
        (cond_df["account_type"] == t)
        & (cond_df["alpha"] == REPRESENTATIVE_ALPHA)
        & (cond_df["stage"].isin(["global", "stratified"]))
    ]["ter"].to_numpy()
    bin_edges = np.linspace(cell_ter.min(), cell_ter.max(), 31)
    for stage in ["global", "stratified"]:
        sub = cond_df[
            (cond_df["account_type"] == t)
            & (cond_df["alpha"] == REPRESENTATIVE_ALPHA)
            & (cond_df["stage"] == stage)
        ]
        color, _, _ = stage_style[stage]
        ax.hist(
            sub["ter"],
            bins=bin_edges,
            alpha=0.5,
            color=color,
            label=f"{stage_label[stage]} (SD = {sub['ter'].std(ddof=0):.2f})",
        )
    ax.axvline(1.0, **REFERENCE_STYLE)
    ax.set_xlabel(rf"Cell TER ($\alpha={REPRESENTATIVE_ALPHA}$)")
    ax.set_ylabel("Hour-of-week cells")
    ax.legend(fontsize=5.8, loc="upper right")
    light_grid(ax, axis="y")
    save_fig(fig, f"{FIG_CELLS_PREFIX}_{t}")
    plt.close(fig)

print(f"\nsaved {OUT_ALL}")
print(f"saved {OUT_TER}")
print(f"saved {OUT_COND_TER}")
for t in ["human", "machine"]:
    print(f"saved {FIG_DIR / (FIG_MARGINAL_PREFIX + '_' + t + '.pdf')}")
    print(f"saved {FIG_DIR / (FIG_CELLS_PREFIX + '_' + t + '.pdf')}")
