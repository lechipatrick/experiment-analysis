import pandas as pd

from experiment_analysis.base.base_randomization_inference import (
    BaseRandomizationInference,
)
from experiment_analysis.constants import CONTROL, METRIC, TREATMENT, VARIATION


class AdditiveMetricRandomizationInference(BaseRandomizationInference):
    def _validate_data_columns(self) -> None:
        if not set({METRIC, VARIATION}).issubset(set(self.data.columns)):
            raise ValueError(
                f"data must contain columns {METRIC} and {VARIATION}"
            )

    def estimate_treatment_effect(self, data: pd.DataFrame) -> float:
        control_mean = data[data[VARIATION] == CONTROL][METRIC].mean()
        treatment_mean = data[data[VARIATION] == TREATMENT][METRIC].mean()
        return treatment_mean - control_mean  # type: ignore
