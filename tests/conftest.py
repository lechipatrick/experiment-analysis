import pandas as pd
import pytest

from experiment_analysis.constants import (
    CONTROL,
    METRIC,
    METRIC_DENOMINATOR,
    METRIC_NUMERATOR,
    TREATMENT,
    VARIATION,
)


@pytest.fixture
def test_data_additive() -> pd.DataFrame:
    data = {
        METRIC: [1, 2, 3, 4],
        VARIATION: [CONTROL, CONTROL, TREATMENT, TREATMENT],
    }
    df = pd.DataFrame.from_dict(data)
    return df


@pytest.fixture
def test_data_ratio() -> pd.DataFrame:
    data = {
        METRIC_NUMERATOR: [1, 2, 3, 4],
        METRIC_DENOMINATOR: [5, 6, 7, 8],
        VARIATION: [CONTROL, CONTROL, TREATMENT, TREATMENT],
    }
    df = pd.DataFrame.from_dict(data)
    return df
