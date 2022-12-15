import numpy as np
import pandas as pd
import pytest

from experiment_analysis.constants import CONTROL, METRIC, TREATMENT, VARIATION


@pytest.fixture
def test_data() -> pd.DataFrame:
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
