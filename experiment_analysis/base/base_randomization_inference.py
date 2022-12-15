from abc import abstractmethod

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from experiment_analysis.constants import CONTROL, TREATMENT, VARIATION


class BaseRandomizationInference:
    """
    An abstract class that outlines the steps that any randomization inference method should follow.
    This class expects the data for analysis to be a pandas DataFrame, with certain required columns.
    The specific requirements would differ for different child classes.
    """

    def __init__(self, *, data: pd.DataFrame, num_randomizations: int) -> None:
        self.data = data
        self.num_randomizations = num_randomizations
        self._control_proportion = None
        self._treatment_effect = None
        self._data_size = None

        self._validate()

        self._randomized_assignment_data = data.copy()

    def _validate(self) -> None:
        self._validate_data_inputs()
        self._validate_variation()

    def _validate_data_inputs(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if (
            not isinstance(self.num_randomizations, int)
            or self.num_randomizations < 0
        ):
            raise ValueError("num_randomizations must be a positive integer")

    def _validate_variation(self) -> None:
        if not set({VARIATION}).issubset(set(self.data.columns)):
            raise ValueError(f"data must contain column {VARIATION}")

        variation_values = set(self.data[VARIATION].unique())
        if set({CONTROL, TREATMENT}) != variation_values:
            raise ValueError(
                f"variation must take values {CONTROL} and {TREATMENT}"
            )

    @property
    def control_proportion(self) -> float:
        if not self._control_proportion:
            variation_counts = self.data[VARIATION].value_counts().to_dict()
            self._control_proportion = variation_counts[CONTROL] / (
                variation_counts[CONTROL] + variation_counts[TREATMENT]
            )
        return self._control_proportion  # type: ignore

    @property
    def data_size(self) -> int:
        if not self._data_size:
            self._data_size = len(self.data)  # type: ignore
        return self._data_size  # type: ignore

    def get_randomized_assignment_data(self) -> pd.DataFrame:
        rv = np.random.uniform(low=0, high=1, size=(self.data_size,))
        randomized_assignment = np.where(
            rv < self.control_proportion, CONTROL, TREATMENT
        )

        self._randomized_assignment_data[VARIATION] = randomized_assignment

        return self._randomized_assignment_data

    @abstractmethod
    def estimate_treatment_effect(self, data: pd.DataFrame) -> float:
        pass

    @property
    def treatment_effect(self) -> float:
        if not self._treatment_effect:
            self._treatment_effect = self.estimate_treatment_effect(self.data)  # type: ignore
        return self._treatment_effect  # type: ignore

    def get_randomized_treatment_effects(self) -> NDArray[np.float64]:
        randomized_treatment_effects = np.zeros((self.num_randomizations,))

        for i in range(self.num_randomizations):
            randomized_assignment_data = self.get_randomized_assignment_data()
            treatment_effect = self.estimate_treatment_effect(
                randomized_assignment_data
            )
            randomized_treatment_effects[i] = treatment_effect

        return randomized_treatment_effects

    def get_p_value(self) -> float:
        randomized_treatment_effects = self.get_randomized_treatment_effects()
        observed_treatment_effect = self.treatment_effect
        p_value = (
            np.abs(randomized_treatment_effects)
            > np.abs(observed_treatment_effect)
        ).mean()
        return p_value  # type: ignore
