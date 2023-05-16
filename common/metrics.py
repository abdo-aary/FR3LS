import numpy as np
from scipy.stats import pearsonr
import warnings


def correlation(TS1: np.ndarray, TS2: np.ndarray) -> float:
    """
    Correlation between two matrices computation = mean of dimensional correlations

    :param TS1: Multidimentional TS
    :param TS2: Multidimentional TS
    :return: correlation coefficient
    """
    warnings.filterwarnings("error")

    mean_correlation = 0
    nbr_corr_columns = 0
    for j in range(TS1.shape[1]):
        try:
            corr, _ = pearsonr(TS1[:, j], TS2[:, j])
            mean_correlation += abs(corr)
            nbr_corr_columns += 1
        except Exception as e:
            pass
    warnings.filterwarnings("default")

    return mean_correlation / nbr_corr_columns
