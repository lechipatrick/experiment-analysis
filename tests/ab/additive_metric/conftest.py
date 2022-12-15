import pandas as pd
import pytest
from experiment_analysis.constants import CONTROL, METRIC, TREATMENT, VARIATION


@pytest.fixture
def test_data() -> pd.DataFrame:
    num_units = 1000

    variation_control = [CONTROL for _ in range(num_units)]
    variation_treatment = [TREATMENT for _ in range(num_units)]

    metric_control = [0 for _ in range(num_units)]
    metric_treatment = [1 for _ in range(num_units)]

    data = {
        METRIC: metric_control + metric_treatment,
        VARIATION: variation_control + variation_treatment,
    }
    df = pd.DataFrame.from_dict(data)
    return df
