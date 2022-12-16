import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

from experiment_analysis.ab.additive_metric.additive_metric import (
    AdditiveMetricInference,
)
from experiment_analysis.constants import METRIC, VARIATION
from experiment_analysis.data_models.additive_metric_data import (
    AdditiveMetricData,
)


def generate_test_data(
    treatment_effect: float = 1, std: float = 1
) -> AdditiveMetricData:
    # generate experiment data corresponding to specified treatment effect and variance
    # if variance is zero, then we get constant treatment effect
    # if variance is non-zero, then we get average treatment effect
    num_units = 1000

    variation_control = ["control" for _ in range(num_units)]
    variation_treatment = ["treatment" for _ in range(num_units)]

    metric_control = np.random.normal(loc=0, scale=std, size=(num_units,))
    metric_treatment = np.random.normal(
        loc=treatment_effect, scale=std, size=(num_units,)
    )

    data = {
        METRIC: np.hstack((metric_control, metric_treatment)),
        VARIATION: variation_control + variation_treatment,
    }
    df = pd.DataFrame.from_dict(data)
    return AdditiveMetricData(data=df)


def test_get_control_proportion() -> None:
    test_data = generate_test_data(treatment_effect=1, std=1)
    assert (
        AdditiveMetricInference.get_control_proportion(test_data.data) == 0.5
    )


def test_estimate_treatment_effect_additive_metric() -> None:
    test_data = generate_test_data(1, 0)
    assert (
        AdditiveMetricInference.estimate_treatment_effect(test_data.data)
        == 1.0
    )


def test_treatment_effect() -> None:
    test_data = generate_test_data(1, 0)
    assert AdditiveMetricInference(test_data).treatment_effect == 1.0


def test_p_value() -> None:
    test_data = generate_test_data(1, 0)
    inference = AdditiveMetricInference(test_data)
    assert (
        inference.get_p_value(method="bootstrap", num_bootstraps=1000) < 0.01
    )
    assert (
        inference.get_p_value(method="randomization", num_randomizations=1000)
        < 0.01
    )

def test_p_value_distribution_z_test() -> None:

    """
    If the null is indeed true, p-values should follow a uniform distribution. Furthermore, if we use a 0.05
    threshold to reject the null, then the fraction of such rejections should be around 5 percent
    This test covers the randomization and bootstrap approaches, both of which are slow
    """
    # generate the p-values
    num_sims = 1000
    p_values = np.zeros((num_sims,))

    for i in range(num_sims):
        if i % 10 == 0:
            print(f"at iteration {i}")
        test_data = generate_test_data(treatment_effect=0, std=1)
        inference = AdditiveMetricInference(test_data)
        p_value_z_test = inference.get_p_value(
            method="ztest"
        )
        p_values[i] = p_value_z_test

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



@pytest.mark.slow
def test_p_value_distribution() -> None:

    """
    If the null is indeed true, p-values should follow a uniform distribution. Furthermore, if we use a 0.05
    threshold to reject the null, then the fraction of such rejections should be around 5 percent
    This test covers the randomization and bootstrap approaches, both of which are slow
    """
    # generate the p-values
    num_sims = 1000
    p_values_bootstrap = np.zeros((num_sims,))
    p_values_randomization = np.zeros((num_sims,))

    for i in range(num_sims):
        if i % 10 == 0:
            print(f"at iteration {i}")
        test_data = generate_test_data(treatment_effect=0, std=1)
        inference = AdditiveMetricInference(test_data)
        p_value_bootstrap = inference.get_p_value(
            method="bootstrap", num_bootstraps=1000
        )
        p_value_randomization = inference.get_p_value(
            method="randomization", num_randomizations=1000
        )
        p_values_bootstrap[i] = p_value_bootstrap
        p_values_randomization[i] = p_value_randomization

    # fpr should be around 5%
    for p_values in [p_values_bootstrap, p_values_randomization]:
        fpr = np.where(p_values < 0.05, 1, 0).mean()
        assert 0.04 < fpr < 0.06

    # p values should be uniformly distributed
    num_buckets = 20
    for p_values in [p_values_bootstrap, p_values_randomization]:
        f_obs = np.zeros((num_buckets,))
        for i in range(0, 20):
            start = i * 0.05
            end = (i + 1) * 0.05
            f = np.where((p_values >= start) & (p_values < end), 1, 0).sum()
            f_obs[i] = f

        _, p = chisquare(f_obs)
        assert p > 0.05
