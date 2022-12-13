import numpy as np
from scipy.stats import chisquare

from experiment_analysis.ab.randomization_inference import (
    RandomizationInference,
)


class TestRandomizationInference:
    """
    If the null is indeed true, p-values should follow a uniform distribution. Furthermore, if we use a 0.05
    threshold to reject the null, then the fraction of such rejections should be around 5 percent
    """
    def test_fpr_5_percent(self):
        pvalues = []
        for _ in range(1000):
            control = np.random.normal(0, 1, size=(1000,))
            treatment = np.random.normal(0, 1, size=(1000,))
            rand_inf = RandomizationInference(
                control=control,
                treatment=treatment,
                num_randomizations=1000,
            )
            p_value = rand_inf.get_p_value()
            pvalues.append(p_value)
        pvalues = np.array(pvalues)
        fpr = np.where(pvalues< 0.05, 1, 0).mean()
        print(fpr)
        assert 0.04 < fpr < 0.06

    def test_p_values_uniformly_distributed(self):
        """
        This test makes a stronger assertion: p-values should be distributed uniformly under the null.
        A chisquare test is used.
        (A statistical procedure to verify correctness of another statistical procedure! Recursion in statistics?)
        """
        # generate the p-values
        pvalues = []
        for _ in range(1000):
            control = np.random.normal(0, 1, size=(1000,))
            treatment = np.random.normal(0, 1, size=(1000,))
            rand_inf = RandomizationInference(
                control=control,
                treatment=treatment,
                num_randomizations=1000,
            )
            p_value = rand_inf.get_p_value()
            pvalues.append(p_value)
        pvalues = np.array(pvalues)

        # generate the empirical frequencies
        f_obs = []
        for i in range(0, 20):
            start = i * 0.05
            end = (i + 1) * 0.05
            f = np.where((pvalues >= start) & (pvalues < end), 1, 0).sum()
            f_obs.append(f)

        f_obs = np.array(f_obs)
        chisq, p = chisquare(f_obs)
        assert p > 0.05
