import numpy as np
from scipy.stats import nbinom


P_FLOOR = 1e-300


def nb_zero_probability(mean, size):
    return (size / (size + mean)) ** size


def match_nb_mean(target_mean, size):
    lower = np.full_like(target_mean, 1e-6, dtype=float)
    upper = np.full_like(target_mean, 1.0, dtype=float)

    # upper search bound
    for _ in range(80):
        active_mean = upper / (1 - nb_zero_probability(upper, size))
        below_target = active_mean < target_mean
        if not below_target.any():
            break
        upper = np.where(below_target, upper * 2, upper)

    # bisection
    for _ in range(50):
        midpoint = 0.5 * (lower + upper)
        active_mean = midpoint / (1 - nb_zero_probability(midpoint, size))
        lower = np.where(active_mean < target_mean, midpoint, lower)
        upper = np.where(active_mean < target_mean, upper, midpoint)

    return 0.5 * (lower + upper)


def zt_nb_midp(count, mean, size):
    probability = size / (size + mean)
    p_zero = nbinom.pmf(0, size, probability)
    p_mid = (
        nbinom.sf(count - 1, size, probability)
        - 0.5 * nbinom.pmf(count, size, probability)
    ) / (1 - p_zero)
    return np.clip(p_mid, P_FLOOR, 1.0)


def zt_nb_negloglik(size, count, target_mean):
    mean = match_nb_mean(target_mean, size)
    probability = size / (size + mean)
    p_zero = nbinom.pmf(0, size, probability)
    log_likelihood = (
        nbinom.logpmf(count, size, probability)
        - np.log1p(-p_zero)
    )
    return -log_likelihood.sum()
