from typing import Callable

import numpy as np
from numpy.typing import NDArray


class Randomization:
    """
    holds various randomization-related statistical procedures
    for speed, operations are limited to numpy arrays
    """

    def __init__(self, data: NDArray[np.float64]) -> None:
        self.data = data
        self._randomized_data = data.copy()
        self.size = len(data)
        self.control_proportion = 1 - data[:, -1].mean()

    def get_simple_random_assignment(self) -> NDArray[np.int64]:
        # returns a numpy array of integers, with values of 0, 1
        # value of 0 corresponds to "control" and value of 1 corresponds to "treatment"
        rv = np.random.uniform(low=0, high=1, size=self.size)
        random_assignment = np.where(rv < self.control_proportion, 0, 1)
        return random_assignment

    def get_simple_randomized_assignment_data(self) -> NDArray[np.int64]:
        self._randomized_data[:, -1] = self.get_simple_random_assignment()
        return self._randomized_data  # type: ignore

    def get_simple_randomized_assignment_estimates(
        self,
        estimation_func: Callable[[NDArray[np.float64]], float],
        num_randomizations: int,
    ) -> NDArray[np.float64]:

        estimates = np.zeros(num_randomizations)

        for i in range(num_randomizations):
            randomized_assignment_data = (
                self.get_simple_randomized_assignment_data()
            )
            estimate = estimation_func(randomized_assignment_data)  # type: ignore
            estimates[i] = estimate

        return estimates

    @classmethod
    def get_p_value(
        cls,
        estimate: float,
        random_assignment_estimates: NDArray[np.float64],
    ) -> float:
        p_value = (
            np.abs(random_assignment_estimates) > np.abs(estimate)
        ).mean()
        return p_value  # type: ignore
