from typing import Optional

import pandas as pd
from pydantic import BaseModel
from pydantic.class_validators import validator

from experiment_analysis.constants import CONTROL, METRIC, TREATMENT, VARIATION


class AdditiveMetricData(BaseModel):
    data: pd.DataFrame
    randomization_unit: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("data")
    def data_contains_columns(cls, data: pd.DataFrame) -> pd.DataFrame:
        if not set({METRIC, VARIATION}).issubset(set(data.columns)):

            raise ValueError(f"data must contain columns {METRIC, VARIATION}")
        return data

    @validator("data")
    def data_variation_column_contains_values(
        cls, data: pd.DataFrame
    ) -> pd.DataFrame:
        variation_values = set(data[VARIATION].unique())
        if set({CONTROL, TREATMENT}) != variation_values:
            raise ValueError(
                f"variation must take values {CONTROL} and {TREATMENT}"
            )
        return data
