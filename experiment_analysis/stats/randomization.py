from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from experiment_analysis.constants import CONTROL, TREATMENT, VARIATION


class Randomization:
    @classmethod
    def simple_randomize_assignment_data(
        cls, data: pd.DataFrame, control_proportion: float
    ) -> pd.DataFrame:
        rv = np.random.uniform(low=0, high=1, size=(len(data),))
        randomized_assignment = np.where(
            rv < control_proportion, CONTROL, TREATMENT
        )

        randomized_assignment_data = data.copy()
        randomized_assignment_data[VARIATION] = randomized_assignment

        return randomized_assignment_data

    @classmethod
    def get_simple_randomized_assignment_estimates(
        cls,
        data: pd.DataFrame,
        estimation_func: Callable[[pd.DataFrame], Any],
        control_proportion: float,
        num_randomizations: int,
    ) -> NDArray[np.float64]:

        randomized_assignment_estimates = np.zeros((num_randomizations,))

        for i in range(num_randomizations):
            randomized_assignment_data = cls.simple_randomize_assignment_data(
                data, control_proportion
            )
            estimate = estimation_func(randomized_assignment_data)
            randomized_assignment_estimates[i] = estimate

        return randomized_assignment_estimates

    @classmethod
    def cluster_randomize_assignment_data(
        cls, data: pd.DataFrame, cluster_col: str
    ) -> pd.DataFrame:
        # randomize assignment status at the level of clusters instead of at the level of units
        # need a cluster identifier in the data
        pass

    @classmethod
    def get_p_value(
        cls,
        estimate: float,
        randomized_assignment_estimates: NDArray[np.float64],
    ) -> float:
        p_value = (
            np.abs(randomized_assignment_estimates) > np.abs(estimate)
        ).mean()
        return p_value  # type: ignore
