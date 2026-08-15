from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


SCORES_FILE = Path("outputs/tables/04_human_poisson_scores.parquet")
SIMULATION_SIZE = 2_000_000
ALPHAS = [0.05, 0.01, 0.005, 0.001]
P_FLOOR = 1e-300

rng = np.random.default_rng(0)

scores = pd.read_parquet(SCORES_FILE, columns=["lambda_hat"])
rates = scores["lambda_hat"].to_numpy()

# sampled generating rates
rates = rates[rng.integers(0, len(rates), SIMULATION_SIZE)]
counts = rng.poisson(rates)

# active-window sample
active = counts > 0
counts = counts[active]
rates = rates[active]


def zt_pvalue(count, rate):
    p_nonzero = -np.expm1(-rate)
    return np.clip(
        poisson.sf(count - 1, rate) / p_nonzero,
        P_FLOOR,
        1.0,
    )


def zt_midp(count, rate):
    p_nonzero = -np.expm1(-rate)
    p_mid = (
        poisson.sf(count - 1, rate)
        - 0.5 * poisson.pmf(count, rate)
    ) / p_nonzero
    return np.clip(p_mid, P_FLOOR, 1.0)


p_standard = zt_pvalue(counts, rates)
p_mid = zt_midp(counts, rates)

print(f"simulated active windows: {len(p_standard):,}")

print("standard p:")
for alpha in ALPHAS:
    print(f"  alpha={alpha:.3f}  TER={(p_standard < alpha).mean() / alpha:.3f}")

print("mid-p:")
for alpha in ALPHAS:
    print(f"  alpha={alpha:.3f}  TER={(p_mid < alpha).mean() / alpha:.3f}")
