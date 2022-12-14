import numpy as np
from scipy.stats import chisquare

from experiment_analysis.ab.additive_metric.randomization_inference import (
    RandomizationInference,
)


class TestRandomizationInference:
    """
    If the null is indeed true, p-values should follow a uniform distribution. Furthermore, if we use a 0.05
    threshold to reject the null, then the fraction of such rejections should be around 5 percent
    """
    def test_p_value(self) -> None:
        """
        This test makes a stronger assertion: p-values should be distributed uniformly under the null.
        A chisquare test is used.
        (A statistical procedure to verify correctness of another statistical procedure! Recursion in statistics?)
        """
        # generate the p-values
        num_sims = 1000
        pvalues = np.zeros((num_sims,))
        for i in range(num_sims):
            control = np.random.normal(0, 1, size=(1000,))
            treatment = np.random.normal(0, 1, size=(1000,))
            rand_inf = RandomizationInference(
                control=control,
                treatment=treatment,
                num_randomizations=1000,
            )
            p_value = rand_inf.get_p_value()
            pvalues[i] = p_value

        # fpr around 5%
        fpr = np.where(pvalues < 0.05, 1, 0).mean()
        assert 0.04 < fpr < 0.06

        # p value uniformly distributed
        # generate the empirical frequencies
        num_buckets = 20
        f_obs = np.zeros((num_buckets, ))
        for i in range(0, 20):
            start = i * 0.05
            end = (i + 1) * 0.05
            f = np.where((pvalues >= start) & (pvalues < end), 1, 0).sum()
            f_obs[i] = f

        _, p = chisquare(f_obs)
        assert p > 0.05
