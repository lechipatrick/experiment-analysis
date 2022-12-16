from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


class Bootstrap:
    @classmethod
    def simple_bootstrap_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        return data.sample(frac=1, replace=True, ignore_index=True)

    @classmethod
    def get_simple_bootstrapped_estimates(
        cls,
        data: pd.DataFrame,
        estimation_func: Callable[[pd.DataFrame], Any],
        num_bootstraps: int,
    ) -> NDArray[np.float64]:
        bootstrapped_estimates = np.zeros((num_bootstraps,))

        for i in range(num_bootstraps):
            bootstrapped_data = cls.simple_bootstrap_data(data)
            estimate = estimation_func(bootstrapped_data)
            bootstrapped_estimates[i] = estimate

        return bootstrapped_estimates

    @classmethod
    def cluster_bootstrap_data(
        cls, data: pd.DataFrame, cluster_col: str
    ) -> pd.DataFrame:
        # bootstrap clusters of units, not individual units
        # need a cluster identifier in the data
        pass

    @classmethod
    def get_p_value(
        cls, estimate: float, bootstrapped_estimates: NDArray[np.float64]
    ) -> float:
        se = bootstrapped_estimates.std()
        z_statistic = estimate / se
        p_value = 2 * stats.norm.sf(z_statistic, loc=0, scale=1)
        return p_value  # type: ignore
