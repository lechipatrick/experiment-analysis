# this should contain a class that provides various inference methods for additive metrics
# including randomization, bootstrap, and z-test

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from experiment_analysis.base.base import (
    bootstrap,
    estimate_treatment_effect_additive_metric,
    get_p_value_bootstrap,
    validate_data_columns,
    validate_data_type,
    validate_num_draws,
    validate_variation,
)
from experiment_analysis.constants import METRIC, VARIATION


class AdditiveMetricInference:
    def __init__(self, *, data: pd.DataFrame, num_draws: int = 10000) -> None:
        self.data = data
        self.num_draws = num_draws
        self._treatment_effect = None

        self._validate()

    def _validate(self) -> None:
        validate_data_type(self.data)
        validate_data_columns(self.data, columns=[METRIC, VARIATION])
        validate_num_draws(self.num_draws)
        validate_variation(self.data)

    def get_p_value_bootstrap(self):
        pass

    def get_p_value_randomization(self):
        pass
