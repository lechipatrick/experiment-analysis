from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError, validator
from scipy.stats import chisquare

from experiment_analysis.ab.additive_metric.additive_metric import (
    AdditiveMetricInference,
)
from experiment_analysis.constants import METRIC, VARIATION
from experiment_analysis.ab.additive_metric.additive_metric import AdditiveMetricData
from experiment_analysis.data_models.additive_metric_data import AdditiveMetricData
import pytest

def generate_test_data(treatment_effect: float = 1, std: float = 1):
    # generate experiment data corresponding to specified treatment effect and variance
    # if variance is zero, then we get constant treatment effect
    # if variance is non-zero, then we get average treatment effect
    num_units = 1000

    variation_control = ["control" for _ in range(num_units)]
    variation_treatment = ["treatment" for _ in range(num_units)]

    metric_control = np.random.normal(loc=0, scale=std, size=(num_units,))
    metric_treatment = np.random.normal(loc=treatment_effect, scale=std, size=(num_units,))

    data = {
        METRIC: np.hstack((metric_control, metric_treatment)),
        VARIATION: variation_control + variation_treatment,
    }
    df = pd.DataFrame.from_dict(data)
    return AdditiveMetricData(data=df)


def test_get_control_proportion():
    test_data = generate_test_data(treatment_effect=1, std=1)
    assert AdditiveMetricInference.get_control_proportion(test_data.data) == 0.5

def test_estimate_treatment_effect_additive_metric():
    test_data = generate_test_data(1, 0)
    assert AdditiveMetricInference.estimate_treatment_effect(test_data.data) == 1.0

def test_treatment_effect():
    test_data = generate_test_data(1, 0)
    assert AdditiveMetricInference(test_data).treatment_effect == 1.0

def test_p_value():
    test_data = generate_test_data(1, 0)
    inference = AdditiveMetricInference(test_data)
    assert inference.get_p_value(method="bootstrap", num_bootstraps=1000) < 0.01
    assert inference.get_p_value(method="randomization", num_randomizations=1000) < 0.01

@pytest.mark.slow
def test_p_value_distribution():

    """
    If the null is indeed true, p-values should follow a uniform distribution. Furthermore, if we use a 0.05
    threshold to reject the null, then the fraction of such rejections should be around 5 percent
    """
    # generate the p-values
    num_sims = 100
    pvalues_bootstrap = np.zeros((num_sims,))
    pvalues_randomization = np.zeros((num_sims,))

    for i in range(num_sims):
        if i % 10 == 0:
            print(f"at iteration {i}")
        test_data = generate_test_data(treatment_effect=0, std=1)
        inference = AdditiveMetricInference(test_data)
        p_value_bootstrap = inference.get_p_value(method="bootstrap", num_bootstraps=1000)
        p_value_randomization = inference.get_p_value(method="randomization", num_randomizations=1000)
        pvalues_bootstrap[i] = p_value_bootstrap
        pvalues_randomization[i] = p_value_randomization


    # fpr should be around 5%
    for pvalues in [pvalues_bootstrap, pvalues_randomization]:
        fpr = np.where(pvalues < 0.05, 1, 0).mean()
        assert 0.04 < fpr < 0.06

    # p values should be uniformly distributed
    # generate the empirical frequencies
    num_buckets = 20
    for pvalues in [pvalues_bootstrap, pvalues_randomization]:
        f_obs = np.zeros((num_buckets,))
        for i in range(0, 20):
            start = i * 0.05
            end = (i + 1) * 0.05
            f = np.where((pvalues >= start) & (pvalues < end), 1, 0).sum()
            f_obs[i] = f

        _, p = chisquare(f_obs)
        assert p > 0.05
