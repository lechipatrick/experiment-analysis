from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


class Randomization:
    """
    provides various randomization-related statistical procedures
    for speed, operations are limited to numpy arrays
    """

    @classmethod
    def get_simple_random_assignment(
        cls, size: int, control_proportion: float
    ) -> NDArray[np.int64]:
        # returns a numpy array of integers, with values of 0, 1
        # value of 0 corresponds to "control" and value of 1 corresponds to "treatment"
        rv = np.random.uniform(low=0, high=1, size=(size,))
        random_assignment = np.where(rv < control_proportion, 0, 1)
        return random_assignment

    @classmethod
    def get_simple_random_assignment_estimates(
        cls,
        metric: NDArray[np.float64],
        estimation_func: Callable[[Any], float],
        control_proportion: float,
        num_randomizations: int,
    ) -> NDArray[np.float64]:
        size = len(metric)

        estimates = np.zeros((num_randomizations,))

        for i in range(num_randomizations):
            random_assignment = cls.get_simple_random_assignment(
                size, control_proportion
            )
            estimate = estimation_func(metric, random_assignment)  # type: ignore
            estimates[i] = estimate

        return estimates

    @classmethod
    def get_cluster_random_assignment(cls) -> NDArray[np.float64]:
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
