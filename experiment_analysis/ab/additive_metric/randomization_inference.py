from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class RandomizationInference:
    """
    Provides methods to calculate statistics associated with the randomization inference approach.
    https://jasonkerwin.com/nonparibus/2017/09/25/randomization-inference-vs-bootstrapping-p-values/


    Attributes
    ----------
    control : numpy.array
        observed metric values for control units
    treatment: numpy.array
        observed metric values for treatment units
    num_randomizations : int
        number of randomizations to use to construct empirical distribution of estimates (under the null)

    Methods
    -------
    get_p_value
        returns the p-value
    get_treatment_effect
        returns the observed treatment effect

    """

    def __init__(
        self,
        *,
        control: NDArray[np.float64],
        treatment: NDArray[np.float64],
        num_randomizations: int = 10000,
    ) -> None:
        """
        Initialize the instance with data.

        Parameters
        ----------
        control : numpy.array
            observed metric values for control units
        treatment: numpy.array
            observed metric values for treatment units
        num_randomizations : int
            number of randomizations to use to construct empirical distribution of estimates (under the null)
        """

        self._control = control
        self._treatment = treatment
        self._num_randomizations = num_randomizations
        self._data = np.hstack((control, treatment))
        self._data_length = len(self._data)
        self._control_proportion = len(control) / (
            len(control) + len(treatment)
        )

    @property
    def control(self) -> NDArray[np.float64]:
        return self._control

    @property
    def treatment(self) -> NDArray[np.float64]:
        return self._treatment

    @property
    def num_randomizations(self) -> int:
        return self._num_randomizations

    def get_p_value(self) -> float:
        """
        Compute the p-value for the test against the null hypothesis that the treatment effect is zero.

        Notes
        -----
        The method works by re-assigning observations to treatment/control, and re-compute the treatment effect
        from such random assignment. Over a large number of such randomizations, the treatment effects should center
        around zero with some empirical distribution.
        p-value is obtained from comparing the observed treatment effect against this empirical distribution.

        """
        random_treatment_effects = self._get_random_treatment_effects()
        observed_treatment_effect = self.get_treatment_effect()
        p_value = (
            np.abs(random_treatment_effects)
            > np.abs(observed_treatment_effect)
        ).mean()
        return p_value  # type: ignore

    def get_treatment_effect(self) -> np.float64:
        return np.mean(self.treatment) - np.mean(self.control)

    def _get_random_treatment_effect(
        self,
        random_control: NDArray[np.float64],
        random_treatment: NDArray[np.float64],
    ) -> np.float64:
        return np.mean(random_treatment) - np.mean(random_control)

    def _shuffle(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        assignment = np.random.uniform(0, 1, self._data_length)

        random_control = self._data[assignment < self._control_proportion]
        random_treatment = self._data[assignment >= self._control_proportion]

        return random_control, random_treatment

    def _get_random_treatment_effects(self) -> NDArray[np.float64]:
        random_treatment_effects = []
        for _ in range(self.num_randomizations):
            random_control, random_treatment = self._shuffle()
            treatment_effect = self._get_random_treatment_effect(
                random_control, random_treatment
            )
            random_treatment_effects.append(treatment_effect)

        return np.array(random_treatment_effects)
