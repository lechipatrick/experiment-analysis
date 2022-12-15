from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

from experiment_analysis.ab.additive_metric.additive_metric_randomization_inference import (
    AdditiveMetricRandomizationInference,
)
from experiment_analysis.constants import METRIC, VARIATION


class TestRandomizationInference:
    def test_invalid_data_inputs(self, test_data: Any) -> None:
        with pytest.raises(
            ValueError, match="data must be a pandas DataFrame"
        ):
            _ = AdditiveMetricRandomizationInference(
                data="invalid", num_draws=100
            )

        with pytest.raises(
            ValueError, match="num_draws must be a positive integer"
        ):
            _ = AdditiveMetricRandomizationInference(
                data=test_data, num_draws=-100
            )

        with pytest.raises(ValueError, match="data must contain columns"):
            _ = AdditiveMetricRandomizationInference(
                data=test_data[[METRIC]], num_draws=100
            )

        invalid_test_data = test_data
        invalid_test_data[VARIATION] = "invalid_variation"
        with pytest.raises(ValueError, match="variation must take values"):
            _ = AdditiveMetricRandomizationInference(
                data=invalid_test_data, num_draws=100
            )

    def test_treatment_effect(self, test_data: Any) -> None:
        rand_inf = AdditiveMetricRandomizationInference(
            data=test_data, num_draws=1000
        )
        assert rand_inf.treatment_effect > 0.5

    def test_draw_treatment_effects(self, test_data: Any) -> None:
        # drawn treatment effects should center around zero even if the true treatment effect is non-zero.
        rand_inf = AdditiveMetricRandomizationInference(
            data=test_data, num_draws=10000
        )
        drawn_treatment_effects = rand_inf.draw_treatment_effects()
        assert -0.01 < drawn_treatment_effects.mean() < 0.01

    @pytest.mark.slow
    def test_p_value(self) -> None:
        """
        If the null is indeed true, p-values should follow a uniform distribution. Furthermore, if we use a 0.05
        threshold to reject the null, then the fraction of such rejections should be around 5 percent
        """
        # generate the p-values
        num_sims = 100  # 00
        pvalues = np.zeros((num_sims,))

        _rv = np.random.normal(0, 1, size=(2000,))
        variation = np.where(_rv > 0.5, "treatment", "control")

        for i in range(num_sims):
            if i % 10 == 0:
                print(f"at iteration {i}")
            metric = np.random.normal(0, 1, size=(2000,))
            data = {METRIC: metric, VARIATION: variation}
            df = pd.DataFrame.from_dict(data)
            rand_inf = AdditiveMetricRandomizationInference(
                data=df, num_draws=1000
            )
            p_value = rand_inf.get_p_value()
            pvalues[i] = p_value

        # fpr should be around 5%
        fpr = np.where(pvalues < 0.05, 1, 0).mean()
        assert 0.04 < fpr < 0.06

        # p values should be uniformly distributed
        # generate the empirical frequencies
        num_buckets = 20
        f_obs = np.zeros((num_buckets,))
        for i in range(0, 20):
            start = i * 0.05
            end = (i + 1) * 0.05
            f = np.where((pvalues >= start) & (pvalues < end), 1, 0).sum()
            f_obs[i] = f

        _, p = chisquare(f_obs)
        assert p > 0.05
