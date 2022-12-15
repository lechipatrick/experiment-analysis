import pandas as pd

from experiment_analysis.base.base_randomization_inference import (
    BaseRandomizationInference,
)
from experiment_analysis.constants import CONTROL, METRIC, TREATMENT, VARIATION


class AdditiveMetricBootstrapInference(BaseRandomizationInference):
    def __init__(self, *, data: pd.DataFrame, num_draws: int) -> None:
        super().__init__(data=data, num_draws=num_draws)
        self._validate_data_columns()

    def _validate_data_columns(self) -> None:
        if not set({METRIC, VARIATION}).issubset(set(self.data.columns)):
            raise ValueError(
                f"data must contain columns {METRIC} and {VARIATION}"
            )

    def estimate_treatment_effect(self, data: pd.DataFrame) -> float:
        control_mean = data[data[VARIATION] == CONTROL][METRIC].mean()
        treatment_mean = data[data[VARIATION] == TREATMENT][METRIC].mean()
        return treatment_mean - control_mean  # type: ignore
