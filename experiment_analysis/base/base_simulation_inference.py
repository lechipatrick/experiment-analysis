from abc import abstractmethod

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from experiment_analysis.constants import CONTROL, TREATMENT, VARIATION


class BaseSimulationInference:
    def __init__(self, *, data: pd.DataFrame, num_draws: int) -> None:
        self.data = data
        self.num_draws = num_draws
        self._control_proportion = None
        self._treatment_effect = None
        self._data_size = None

        self._validate()

    def _validate(self) -> None:
        self._validate_data_inputs()
        self._validate_variation()

    def _validate_data_inputs(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if not isinstance(self.num_draws, int) or self.num_draws < 0:
            raise ValueError("num_draws must be a positive integer")

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

    @abstractmethod
    def draw_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def estimate_treatment_effect(self, data: pd.DataFrame) -> float:
        pass

    @property
    def treatment_effect(self) -> float:
        if not self._treatment_effect:
            self._treatment_effect = self.estimate_treatment_effect(self.data)  # type: ignore
        return self._treatment_effect  # type: ignore

    def draw_treatment_effects(self) -> NDArray[np.float64]:
        drawn_treatment_effects = np.zeros((self.num_draws,))

        for i in range(self.num_draws):
            drawn_data = self.draw_data()
            treatment_effect = self.estimate_treatment_effect(drawn_data)
            drawn_treatment_effects[i] = treatment_effect

        return drawn_treatment_effects
