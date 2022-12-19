import numpy as np
import pandas as pd
from scipy.stats import chisquare
from tqdm import tqdm

from experiment_analysis.ab.additive_metric.additive_metric import (
    AdditiveMetricInference,
)
from experiment_analysis.constants import METRIC, VARIATION
from experiment_analysis.data_models.additive_metric_data import (
    AdditiveMetricData,
)


def generate_test_data(
    num_units: int,
    treatment_effect: float,
    std: float,
    control_proportion: float = 0.2,
) -> AdditiveMetricData:
    # generate experiment data corresponding to specified treatment effect and variance
    # if variance is zero, then we get constant treatment effect
    # if variance is non-zero, then we get average treatment effect
    # 1/5 control, 4/5 treatment
    # TODO: figure out how to do this as a pytest.fixture

    control_num_units = int(num_units * control_proportion)
    treatment_num_units = int(num_units * (1 - control_proportion))

    variation_control = ["control" for _ in range(control_num_units)]
    variation_treatment = ["treatment" for _ in range(treatment_num_units)]

    metric_control = np.random.normal(
        loc=0, scale=std, size=(control_num_units,)
    )
    metric_treatment = np.random.normal(
        loc=treatment_effect, scale=std, size=(treatment_num_units,)
    )

    data = {
        METRIC: np.hstack((metric_control, metric_treatment)),
        VARIATION: variation_control + variation_treatment,
    }
    df = pd.DataFrame.from_dict(data)
    return AdditiveMetricData(data=df)


def test_control_proportion() -> None:
    test_data = generate_test_data(num_units=1000, treatment_effect=1, std=1)
    assert (
        np.abs(AdditiveMetricInference(test_data).control_proportion - 0.2)
        < 0.00001
    )


def test_estimate_treatment_effect() -> None:
    test_data = generate_test_data(num_units=1000, treatment_effect=1, std=0)
    assert AdditiveMetricInference(test_data).treatment_effect == 1.0


def test_assignment() -> None:
    test_data = generate_test_data(num_units=1000, treatment_effect=1, std=0)
    assert AdditiveMetricInference(test_data).assignment.mean() == 0.8


def test_p_value_when_treatment_effect_large() -> None:
    test_data = generate_test_data(num_units=1000, treatment_effect=1, std=0)
    inference = AdditiveMetricInference(test_data)
    # assert (
    #     inference.get_p_value(method="bootstrap", num_bootstraps=1000) < 0.01
    # )
    assert (
        inference.get_p_value(method="randomization", num_randomizations=1000)
        < 0.01
    )


def assert_p_values_under_null(
    method: str, num_units: int, num_sims: int, *args: int, **kwargs: int
) -> None:
    p_values = np.zeros((num_sims,))
    for i in tqdm(range(num_sims)):
        test_data = generate_test_data(
            num_units=num_units, treatment_effect=0, std=1
        )
        inference = AdditiveMetricInference(test_data)
        p_value = inference.get_p_value(method, *args, **kwargs)
        p_values[i] = p_value

    # fpr should be around 5%
    fpr = np.where(p_values < 0.05, 1, 0).mean()
    assert 0.04 < fpr < 0.06

    # p values should be uniformly distributed
    num_buckets = 20
    f_obs = np.zeros((num_buckets,))
    for i in range(0, 20):
        start = i * 0.05
        end = (i + 1) * 0.05
        f = np.where((p_values >= start) & (p_values < end), 1, 0).sum()
        f_obs[i] = f

    _, p = chisquare(f_obs)
    assert p > 0.05


def assert_p_values_under_alternative(
    method: str, num_sims: int, *args: int, **kwargs: int
) -> None:
    p_values = np.zeros((num_sims,))
    for i in tqdm(range(num_sims)):
        # this test_data should give 80% power
        test_data = generate_test_data(
            num_units=1570 * 2,
            treatment_effect=1,
            std=10,
            control_proportion=0.5,
        )
        inference = AdditiveMetricInference(test_data)
        p_value = inference.get_p_value(method, *args, **kwargs)
        p_values[i] = p_value

    # fraction of significance findings should be around 0.8
    detection_rate = np.where(p_values < 0.05, 1, 0).mean()
    assert 0.75 < detection_rate < 0.85


def test_p_value_distribution_z_test_under_null() -> None:
    assert_p_values_under_null(method="ztest", num_units=1000, num_sims=10000)


def test_p_value_distribution_randomization_under_null() -> None:
    assert_p_values_under_null(
        method="randomization",
        num_units=1000,
        num_sims=1000,
        num_randomizations=1000,
    )


def test_p_value_distribution_randomization_under_alternative() -> None:
    assert_p_values_under_alternative(
        method="randomization", num_sims=1000, num_randomizations=1000
    )
