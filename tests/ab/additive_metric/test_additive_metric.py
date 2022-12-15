from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

from experiment_analysis.ab.additive_metric.additive_metric import (
    AdditiveMetricInference
)
from experiment_analysis.constants import METRIC, VARIATION


class TestRandomizationInference:
    def test_invalid_data_inputs(self, test_data: Any) -> None:
        with pytest.raises(
            ValueError, match="data must be a pandas DataFrame"
        ):
            _ = AdditiveMetricInference(
                data="invalid", num_draws=100
            )

        with pytest.raises(
            ValueError, match="num_draws must be a positive integer"
        ):
            _ = AdditiveMetricInference(
                data=test_data, num_draws=-100
            )

        with pytest.raises(ValueError, match="data must contain columns"):
            _ = AdditiveMetricInference(
                data=test_data[[METRIC]], num_draws=100
            )

        invalid_test_data = test_data
        invalid_test_data[VARIATION] = "invalid_variation"
        with pytest.raises(ValueError, match="variation must take values"):
            _ = AdditiveMetricInference(
                data=invalid_test_data, num_draws=100
            )
