from typing import List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from experiment_analysis.constants import CONTROL, METRIC, TREATMENT, VARIATION


def get_control_proportion(data: pd.DataFrame) -> float:
    variation_counts = data[VARIATION].value_counts().to_dict()
    control_proportion = variation_counts[CONTROL] / (
        variation_counts[CONTROL] + variation_counts[TREATMENT]
    )
    return control_proportion  # type: ignore


def randomize_assignment(
    data: pd.DataFrame, control_proportion: float
) -> pd.DataFrame:
    rv = np.random.uniform(low=0, high=1, size=(len(data),))
    randomized_assignment = np.where(
        rv < control_proportion, CONTROL, TREATMENT
    )

    randomized_assignment_data = data.copy()
    randomized_assignment_data[VARIATION] = randomized_assignment

    return randomized_assignment_data


def bootstrap(data: pd.DataFrame) -> pd.DataFrame:
    return data.sample(frac=1, replace=True, ignore_index=True)


def get_p_value_bootstrap(
    observed_treatment_effect: float,
    drawn_treatment_effects: NDArray[np.float64],
) -> float:
    se = drawn_treatment_effects.std()
    z_statistic = observed_treatment_effect / se
    p_value = 2 * stats.norm.sf(z_statistic, loc=0, scale=1)
    return p_value


def get_p_value_randomized_inference(
    observed_treatment_effect: float,
    drawn_treatment_effects: NDArray[np.float64],
) -> float:
    p_value = (
        np.abs(drawn_treatment_effects) > np.abs(observed_treatment_effect)
    ).mean()
    return p_value  # type: ignore


def validate_data_columns(data: pd.DataFrame, columns: List[str]) -> None:
    if not set(columns).issubset(set(data.columns)):
        raise ValueError(f"data must contain columns {columns}")


def validate_data_type(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")


def validate_num_draws(num_draws: int) -> None:
    if not isinstance(num_draws, int) or num_draws < 0:
        raise ValueError("num_draws must be a positive integer")


def validate_variation(data: pd.DataFrame) -> None:
    if not set({VARIATION}).issubset(set(data.columns)):
        raise ValueError(f"data must contain column {VARIATION}")

    variation_values = set(data[VARIATION].unique())
    if set({CONTROL, TREATMENT}) != variation_values:
        raise ValueError(
            f"variation must take values {CONTROL} and {TREATMENT}"
        )
