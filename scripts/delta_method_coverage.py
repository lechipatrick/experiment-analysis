import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.stats import chisquare
from tqdm import tqdm

from experiment_analysis.ab.ratio_metric_inference import RatioMetricInference
from experiment_analysis.constants import (
    METRIC_DENOMINATOR,
    METRIC_NUMERATOR,
    VARIATION,
)
from experiment_analysis.data_models.ratio_metric_data import RatioMetricData


def generate_test_data(
        num_units: int,
        control_mean=np.array([1, 1]),
        cov: NDArray[np.float64] = np.array([[1, 0], [0, 1]]),
        control_proportion: float = 0.5,
) -> RatioMetricData:
    # TODO: figure out how to do this as a pytest.fixture

    control_num_units = int(num_units * control_proportion)
    treatment_num_units = int(num_units * (1 - control_proportion))

    treatment_mean = control_mean

    variation_control = ["control" for _ in range(control_num_units)]
    variation_treatment = ["treatment" for _ in range(treatment_num_units)]

    (
        metric_control_numerator,
        metric_control_denominator,
    ) = np.random.multivariate_normal(control_mean, cov, control_num_units).T
    (
        metric_treatment_numerator,
        metric_treatment_denominator,
    ) = np.random.multivariate_normal(
        treatment_mean, cov, treatment_num_units
    ).T

    data = {
        METRIC_NUMERATOR: np.hstack(
            (metric_control_numerator, metric_treatment_numerator)
        ),
        METRIC_DENOMINATOR: np.hstack(
            (metric_control_denominator, metric_treatment_denominator)
        ),
        VARIATION: variation_control + variation_treatment,
    }
    df = pd.DataFrame.from_dict(data)
    return RatioMetricData(data=df)


def simulate(num_units, control_mean, cov, control_proportion=0.5, num_sims=1000):
    c = 0
    p = 0
    for _ in tqdm(range(num_sims)):
        test_data = generate_test_data(num_units, control_mean, cov, control_proportion)
        inference = RatioMetricInference(test_data)
        lower, upper = inference.get_confidence_interval(0.95, "delta")
        if lower < 0 < upper:
            c += 1
        p_value = inference.get_p_value("delta")
        if p_value < 0.05:
            p += 1
    print(f"coverage rate is {c / num_sims}")
    print(f"FPR is {p / num_sims}")


sigma2 = 20
num_units = 1000
mu = 10
simulate(num_units=num_units, control_mean=np.array([.001, mu]), cov=np.array([[1, 0], [0, sigma2]]), num_sims=1000)
se = np.sqrt(sigma2 / (num_units / 2))
print(f"ratio is {1/ se}")
# it looks like coverage can be too high.
# this happens when the ratio of mean to normalized se is too low (< 3)
