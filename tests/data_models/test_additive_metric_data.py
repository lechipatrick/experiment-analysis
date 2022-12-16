from typing import Any

import pytest
from pydantic import ValidationError

from experiment_analysis.data_models.additive_metric_data import (
    AdditiveMetricData,
)


def test_columns_missing(test_data: Any) -> None:
    with pytest.raises(ValidationError, match="data must contain columns"):
        _ = AdditiveMetricData(data=test_data[["metric"]])


def test_variation_groups_missing(test_data: Any) -> None:
    data = test_data
    data["variation"] = "invalid"
    with pytest.raises(ValidationError, match="variation must take values"):
        _ = AdditiveMetricData(data=data)
