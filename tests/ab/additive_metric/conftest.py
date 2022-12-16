import numpy as np
import pandas as pd
import pytest

from experiment_analysis.constants import CONTROL, METRIC, TREATMENT, VARIATION


@pytest.fixture
def test_data() -> pd.DataFrame:
    # generate metric data corresponding to 2000 units, with 50/50 control/treatment and average treatment effect = 1
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
    return df

@pytest.fixture
def test_data_constant_treatment_effect() -> pd.DataFrame:
    # similar to test_data fixture, but fix treatment effect to be constant 1
    num_units = 1000

    variation_control = [CONTROL for _ in range(num_units)]
    variation_treatment = [TREATMENT for _ in range(num_units)]

    metric_control = np.zeros((num_units,))
    metric_treatment = np.ones((num_units,))

    data = {
        METRIC: np.hstack((metric_control, metric_treatment)),
        VARIATION: variation_control + variation_treatment,
    }
    df = pd.DataFrame.from_dict(data)
    return df
