from typing import Any

import pytest
from pydantic import ValidationError

from experiment_analysis.data_models.ratio_metric_data import RatioMetricData


def test_happy_path(test_data_ratio: Any) -> None:
    _ = RatioMetricData(data=test_data_ratio)


def test_columns_missing(test_data_ratio: Any) -> None:
    with pytest.raises(ValidationError, match="data must contain columns"):
        _ = RatioMetricData(data=test_data_ratio[["metric_denominator"]])


def test_variation_groups_missing(test_data_ratio: Any) -> None:
    data = test_data_ratio
    data["variation"] = "invalid"
    with pytest.raises(ValidationError, match="variation must take values"):
        _ = RatioMetricData(data=data)
