"""src/glucose/utils/evaluation.py

This utils file is designed to provide custom
evaluation functions which will be used in our manuscript.

It currently includes:

- rmse_custom
- corr_custom
- crosscorr
- perc
- get_bootstrap_ci

"""

import math
import numpy as np
from tensorflow.keras.metrics import mean_squared_error
from glucose.settings import SEED
import scipy
from typing import Callable


def get_bootstrap_ci(y_true: np.array,
                     y_pred: np.array,
                     bootstraps: int = 1000,
                     func=None) -> np.array:
    """
    Returns bootstrapped 95% confidence interval for a certain
    function (`func`) given y_true and y_pred.

    Parameters
    ----------
    y_true: np.array
        The labels for the instances.
    y_pred
        The predictions for the instances
    bootstraps: int = 1000
        The amount of bootstraps
    func
        The function to apply.

    Returns
    -------
    np.array
        Returns the median and confidence intervals.

    """

    scores = []

    np.random.seed(SEED)

    for i in range(bootstraps):
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue

        scores.append(func(y_true[indices], y_pred[indices]))

    return np.quantile(scores, [0.025, 0.5, 0.975])


def rmse_custom(y_true: np.array,
                y_pred: np.array) -> float:
    """Calculates the root-mean-squared-error (RMSE) of
    given predicted values and labels.

    Parameters
    ----------
    y_true: np.array
        The labels for the instances.
    y_pred: np.array
        The predictions for the instances

    Returns
    -------
    float
        Returns the root-mean-squared error.

    """

    return math.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true: np.array,
              y_pred: np.array):
    """Calculates R^2 of
    given predicted values and labels.

    Parameters
    ----------
    y_true: np.array
        The labels for the instances.
    y_pred: np.array
        The predictions for the instances

    Returns
    -------
    float
        Returns the R^2.

    """

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_true, y_pred)
    return r_value**2


def rho(y_true: np.array,
        y_pred: np.array) -> float:
    """Calculates the Spearman's  Rho of
    given predicted values and labels.

    Parameters
    ----------
    y_true: np.array
        The labels for the instances.
    y_pred: np.array
        The predictions for the instances

    Returns
    -------
    float
        Returns the Spearman's Rho.

    """

    return scipy.stats.spearmanr(y_true, y_pred)[0] ** 2


def crosscorr(y_true: np.array,
              y_pred: np.array) -> float:
    """Calculates the cross-correlation of the
    given predicted and true values.

    Parameters
    ----------
    y_true: np.array
        The labels for the instances.
    y_pred: np.array
        The predictions for the instances

    Returns
    -------
    float
        Returns the cross-correlation coefficient.

    """

    a = (y_true - np.mean(y_true)) / (np.std(y_true) * len(y_true))
    b = (y_pred - np.mean(y_pred)) / np.std(y_pred)
    return np.correlate(a, b)[0]


def within_perc(perc: int) -> Callable[[np.array, np.array], float]:
    """Function to create a wrapper function
    that defines a function which classifies the
    fraction of values that are within the specified
    percentage.

    Parameters
    ----------
    perc: int
        the percentage to classify values within

    Returns
    -------
    Callable[[np.array, np.array], float]
        Returns a function which takes predicted (np.array) and
        true (np.array) values as input and outputs a percentage.

    """

    def wrapper(y_true: np.array,
                y_pred: np.array) -> float:
        """Function to create a wrapper function
        that defines a function which classifies the
        fraction of values that are within the specified
        percentage.

        Parameters
        ----------
        y_true: np.array
            The labels for the instances.
        y_pred: np.array
            The predictions for the instances

        Returns
        -------
        float
            Returns the fraction of values within the percentage.

        """

        _percs = [abs(((r - p) / r) * 100) for r, p in zip(y_true, y_pred)]
        return (sum([1 for p in _percs if p <= perc]) / len(_percs)) * 100

    return wrapper
