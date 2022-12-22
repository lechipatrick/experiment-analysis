from typing import Callable

import numpy as np
from numpy.typing import NDArray

from experiment_analysis.stats.zstatistic import ZStatistic


class Bootstrap:
    """
    holds various bootstrap-related statistical procedures
    for speed, operations are limited to numpy arrays
    """

    def __init__(self, data: NDArray[np.float64], num_bootstraps: int):
        # data should be array of shape (n, k) with the first k-1 columns being metrics, and the last column being
        # assignment
        self.data = data.copy()
        self.num_bootstraps = num_bootstraps
        self.size = len(data)


    def get_simple_bootstrap_data(self) -> NDArray[np.float64]:
        indices = np.random.choice(a=self.size, size=self.size, replace=True)
        return self.data[indices]  # type: ignore

    def get_bootstrap_estimates(
        self,
        estimation_func: Callable[[NDArray[np.float64]], float],
    ) -> NDArray[np.float64]:

        estimates = np.zeros(self.num_bootstraps)

        for i in range(self.num_bootstraps):
            bootstrap_data = self.get_simple_bootstrap_data()
            estimate = estimation_func(bootstrap_data)
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
