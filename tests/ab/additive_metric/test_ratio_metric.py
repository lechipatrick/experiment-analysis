import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.stats import chisquare
from tqdm import tqdm

from experiment_analysis.ab.ratio_metric import RatioMetricInference
from experiment_analysis.constants import (
    METRIC_DENOMINATOR,
    METRIC_NUMERATOR,
    VARIATION,
)
from experiment_analysis.data_models.ratio_metric_data import RatioMetricData


def generate_test_data(
    num_units: int,
    treatment_effect: float,
    cov: NDArray[np.float64] = np.array([[1, 0], [0, 1]]),
    control_proportion: float = 0.5,
) -> RatioMetricData:
    # TODO: figure out how to do this as a pytest.fixture

    control_num_units = int(num_units * control_proportion)
    treatment_num_units = int(num_units * (1 - control_proportion))

    control_mean = np.array([1, 1])
    treatment_mean = np.array([1 + treatment_effect, 1])
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


def test_estimate_treatment_effect() -> None:
    test_data = generate_test_data(
        num_units=1000, treatment_effect=1, cov=np.array([[0, 0], [0, 0]])
    )
    assert RatioMetricInference(test_data).treatment_effect == 1.0


def test_assignment() -> None:
    test_data = generate_test_data(
        num_units=1000, treatment_effect=1, control_proportion=0.2
    )
    assert RatioMetricInference(test_data).assignment.mean() == 0.8


def test_p_value_when_treatment_effect_large() -> None:
    test_data = generate_test_data(num_units=1000, treatment_effect=10)
    inference = RatioMetricInference(test_data)
    # assert inference.get_p_value(method="delta") < 0.01
    assert (
        inference.get_p_value(method="randomization", num_randomizations=1000)
        < 0.01
    )
    assert (
        inference.get_p_value(method="bootstrap", num_bootstraps=1000) < 0.01
    )


def assert_p_value_distribution_under_null(
    method: str, num_units: int, num_sims: int, *args: int, **kwargs: int
) -> None:
    p_values = np.zeros(num_sims)
    for i in tqdm(range(num_sims)):
        test_data = generate_test_data(num_units=num_units, treatment_effect=0)
        inference = RatioMetricInference(test_data)
        p_value = inference.get_p_value(method, *args, **kwargs)
        p_values[i] = p_value

    # fpr should be around 5%
    fpr = np.where(p_values < 0.05, 1, 0).mean()
    print(f"fpr under method {method} is {fpr}")
    try:
        assert 0.03 < fpr < 0.07
    except Exception as exc:
        print("fpr not around 5 percent")
        raise exc

    # p values should be uniformly distributed
    num_buckets = 20
    f_obs = np.zeros(num_buckets)

    for i in range(0, 20):
        start = i * 0.05
        end = (i + 1) * 0.05
        f = np.where((p_values >= start) & (p_values < end), 1, 0).sum()
        f_obs[i] = f

    _, p = chisquare(f_obs)
    try:
        assert p > 0.05
    except Exception as exc:
        print("p_values are not uniformly distribution")
        raise exc


# @pytest.mark.fpr
# def test_p_value_distribution_z_test_under_null() -> None:
#     assert_p_value_distribution_under_null(method="delta", num_units=1000, num_sims=10000)


@pytest.mark.fpr
def test_p_value_distribution_under_null_randomization() -> None:
    assert_p_value_distribution_under_null(
        method="randomization",
        num_units=1000,
        num_sims=2000,
        num_randomizations=1000,
    )


@pytest.mark.fpr
def test_p_value_distribution_under_null_bootstrap() -> None:
    assert_p_value_distribution_under_null(
        method="bootstrap",
        num_units=1000,
        num_sims=2000,
        num_bootstraps=1000,
    )


def assert_p_value_distribution_under_alternative(
    method: str, num_sims: int, *args: int, **kwargs: int
) -> None:
    p_values = np.zeros(num_sims)
    for i in tqdm(range(num_sims)):
        # this test_data should give 80% power at 5% size
        test_data = generate_test_data(
            num_units=3140 * 2,
            treatment_effect=0.1,
            control_proportion=0.5,
        )
        inference = RatioMetricInference(test_data)
        p_value = inference.get_p_value(method, *args, **kwargs)
        p_values[i] = p_value

    # fraction of significance findings should be around 0.8
    detection_rate = np.where(p_values < 0.05, 1, 0).mean()
    print(f"detection rate under method {method} is {detection_rate}")
    assert 0.7 < detection_rate < 0.9


#
#
# # @pytest.mark.power
# # def test_p_value_distribution_z_test_under_alternative() -> None:
# #     assert_p_values_under_alternative(method="delta", num_sims=1000)
#
#
@pytest.mark.power
def test_p_value_distribution_randomization_under_alternative() -> None:
    assert_p_value_distribution_under_alternative(
        method="randomization", num_sims=2000, num_randomizations=1000
    )


#


@pytest.mark.power
def test_p_value_distribution_bootstrap_under_alternative() -> None:
    assert_p_value_distribution_under_alternative(
        method="bootstrap", num_sims=2000, num_bootstraps=1000
    )
