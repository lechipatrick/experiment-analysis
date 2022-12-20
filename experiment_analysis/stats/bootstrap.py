from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from experiment_analysis.stats.zstatistic import ZStatistic


class Bootstrap:
    """
    holds various bootstrap-related statistical procedures
    for speed, operations are limited to numpy arrays
    """

    @classmethod
    def get_simple_bootstrap_data(
        cls, data: NDArray[np.float64], size: int
    ) -> NDArray[np.float64]:
        # data should be array of shape (n, 2) with the first column being metric, and second column being
        # assignment
        # size is the number of observations to draw. usually set to len(data)
        indices = np.random.choice(a=size, size=size, replace=True)
        return data[indices]  # type: ignore

    @classmethod
    def get_simple_bootstrap_estimates(
        cls,
        data: NDArray[np.float64],
        estimation_func: Callable[[Any], Any],
        num_bootstraps: int,
    ) -> NDArray[np.float64]:

        size = len(data)

        estimates = np.zeros(num_bootstraps)

        for i in range(num_bootstraps):
            bootstrap_data = cls.get_simple_bootstrap_data(data, size)
            metric, assignment = bootstrap_data[:, 0], bootstrap_data[:, 1]
            estimate = estimation_func(metric, assignment)  # type: ignore
            estimates[i] = estimate

        return estimates

    @classmethod
    def cluster_bootstrap_data(
        cls, metric: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # bootstrap clusters of units, not individual units
        # need a cluster identifier in the data
        pass

    @classmethod
    def get_p_value(
        cls, estimate: float, bootstrap_estimates: NDArray[np.float64]
    ) -> float:
        se = bootstrap_estimates.std()
        return ZStatistic.get_p_value(estimate, se)
