import numpy as np
import pandas as pd

from experiment_analysis.base.base_simulation_inference import (
    BaseSimulationInference,
)
from experiment_analysis.constants import CONTROL, TREATMENT, VARIATION


class BaseRandomizationInference(BaseSimulationInference):
    def __init__(self, *, data: pd.DataFrame, num_draws: int) -> None:
        super().__init__(data=data, num_draws=num_draws)

        self._randomized_assignment_data = data.copy()

    def draw_data(self) -> pd.DataFrame:
        rv = np.random.uniform(low=0, high=1, size=(self.data_size,))
        randomized_assignment = np.where(
            rv < self.control_proportion, CONTROL, TREATMENT
        )

        self._randomized_assignment_data[VARIATION] = randomized_assignment

        return self._randomized_assignment_data

    def get_p_value(self) -> float:
        drawn_treatment_effects = self.draw_treatment_effects()
        observed_treatment_effect = self.treatment_effect
        p_value = (
            np.abs(drawn_treatment_effects) > np.abs(observed_treatment_effect)
        ).mean()
        return p_value  # type: ignore
