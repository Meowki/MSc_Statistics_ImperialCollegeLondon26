import numpy as np


def empirical_rank_map(fitting_values, target_values):
    fitting_values = np.sort(fitting_values)
    ranks = np.searchsorted(fitting_values, target_values, side="right")
    ranks = np.maximum(ranks, 1)
    return ranks / (len(fitting_values) + 1)
