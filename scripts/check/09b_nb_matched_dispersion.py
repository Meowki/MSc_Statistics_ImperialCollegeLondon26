import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb_utils import match_nb_mean, zt_nb_midp, zt_nb_negloglik
from plotstyle import *

HUMAN_FILE = Path("outputs/tables/25_human_backsolved_scores.parquet")
OUTPUT_FILE = Path("outputs/tables/09b_nb_matched_ter.csv")
FIT_DAYS = (30, 43)
EVAL_DAYS = (44, 57)
ALPHAS = [0.05, 0.01, 0.005, 0.001]
FIT_SAMPLE_SIZE = 1_000_000
SIZE_GRID = np.logspace(np.log10(0.5), np.log10(50), 25)


scores = pd.read_parquet(
    HUMAN_FILE,
    columns=["day", "event_count", "m_hat", "p_mid"],
)

# size fitting
fitting_data = scores[
    (scores["day"] >= FIT_DAYS[0])
    & (scores["day"] <= FIT_DAYS[1])
]
fitting_sample = fitting_data.sample(FIT_SAMPLE_SIZE, random_state=0)
fitting_counts = fitting_sample["event_count"].to_numpy()
fitting_means = fitting_sample["m_hat"].to_numpy()

negative_log_likelihoods = [
    zt_nb_negloglik(size, fitting_counts, fitting_means)
    for size in SIZE_GRID
]
fitted_size = float(SIZE_GRID[np.argmin(negative_log_likelihoods)])
print(f"fitted NB size r = {fitted_size:.3f}")

# matched-mean evaluation
evaluation_data = scores[
    (scores["day"] >= EVAL_DAYS[0])
    & (scores["day"] <= EVAL_DAYS[1])
]
evaluation_counts = evaluation_data["event_count"].to_numpy()
evaluation_means = evaluation_data["m_hat"].to_numpy()
nb_means = match_nb_mean(evaluation_means, fitted_size)
nb_p = zt_nb_midp(evaluation_counts, nb_means, fitted_size)
poisson_p = evaluation_data["p_mid"].to_numpy()

print(f"\neval windows: {len(evaluation_data):,}")
print("alpha    Poisson(25)   NB(matched)   reduction")
rows = []
for alpha in ALPHAS:
    poisson_ter = (poisson_p < alpha).mean() / alpha
    nb_ter = (nb_p < alpha).mean() / alpha
    reduction = (1 - nb_ter / poisson_ter) * 100
    print(
        f"{alpha:<7} {poisson_ter:>10.2f} "
        f"{nb_ter:>12.2f} {reduction:>10.1f}%"
    )
    rows.append({
        "alpha": alpha,
        "poisson_ter": poisson_ter,
        "nb_matched_ter": nb_ter,
        "reduction_pct": reduction,
        "r_hat": fitted_size,
        "eval_windows": len(evaluation_data),
    })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT_FILE, index=False)
print(f"\nsaved {OUTPUT_FILE}")

# tail-exceedance comparison
fig, ax = plt.subplots(figsize=(HALF_W, 2.6))
ax.set_axisbelow(True)
xpos = np.arange(len(result))
width = 0.34

ax.bar(
    xpos - width / 2,
    result["poisson_ter"],
    width,
    color=NAVY,
    label="Poisson",
)
ax.bar(
    xpos + width / 2,
    result["nb_matched_ter"],
    width,
    color=TANGERINE,
    label="Negative binomial",
)
ax.axhline(1.0, **REFERENCE_STYLE)
ax.set_xticks(xpos)
ax.set_xticklabels([str(alpha) for alpha in result["alpha"]])
ax.set_xlabel(r"Nominal threshold $\alpha$")
ax.set_ylabel("TER")
ax.set_yscale("log")
ax.legend(fontsize=7)
light_grid(ax, axis="y")

save_fig(fig, "nb_matched_ter")
plt.close(fig)
print("saved nb_matched_ter")
