import pandas as pd

from experiment_analysis.base.base_randomization_inference import (
    BaseRandomizationInference,
)
from experiment_analysis.base.base_additive_metric import BaseAdditiveMetric
from experiment_analysis.constants import CONTROL, METRIC, TREATMENT, VARIATION


class AdditiveMetricRandomizationInference(BaseRandomizationInference, BaseAdditiveMetric):
    def __init__(self, *, data: pd.DataFrame, num_draws: int) -> None:
        super().__init__(data=data, num_draws=num_draws)
        self._validate_data_columns()

    def _validate_data_columns(self) -> None:
        if not set({METRIC, VARIATION}).issubset(set(self.data.columns)):
            raise ValueError(
                f"data must contain columns {METRIC} and {VARIATION}"
            )

import numpy as np
num_units = 1000

variation_control = [CONTROL for _ in range(num_units)]
variation_treatment = [TREATMENT for _ in range(num_units)]

metric_control = np.random.normal(loc=0, scale=1, size=(num_units,))
metric_treatment = np.random.normal(loc=1, scale=1, size=(num_units,))

data = {
    METRIC: np.hstack((metric_control, metric_treatment)),
    VARIATION: variation_control + variation_treatment,
}
df = pd.DataFrame.from_dict(data)

rand_inf = AdditiveMetricRandomizationInference(data=df, num_draws=100)

rand_inf.treatment_effect