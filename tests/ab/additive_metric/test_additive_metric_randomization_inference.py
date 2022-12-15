import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

from experiment_analysis.ab.additive_metric.additive_metric_randomization_inference import (
    AdditiveMetricRandomizationInference,
)


class TestRandomizationInference:
    @pytest.mark.slow
    def test_p_value(self) -> None:
        """
        If the null is indeed true, p-values should follow a uniform distribution. Furthermore, if we use a 0.05
        threshold to reject the null, then the fraction of such rejections should be around 5 percent
        """
        # generate the p-values
        num_sims = 1000
        pvalues = np.zeros((num_sims,))

        _rv = np.random.normal(0, 1, size=(2000,))
        variation = np.where(_rv > 0.5, "treatment", "control")

        for i in range(num_sims):
            if i % 10 == 0:
                print(f"at iteration {i}")
            metric = np.random.normal(0, 1, size=(2000,))
            data = {"metric": metric, "variation": variation}
            df = pd.DataFrame.from_dict(data)
            rand_inf = AdditiveMetricRandomizationInference(
                data=df, num_randomizations=1000
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
