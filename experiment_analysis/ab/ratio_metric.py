from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from experiment_analysis.ab.additive_metric import AdditiveMetricInference
from experiment_analysis.constants import (
    BOOTSTRAP,
    DELTA,
    FIELLER,
    METRIC_DENOMINATOR,
    METRIC_NUMERATOR,
    RANDOMIZATION,
    TREATMENT,
    VARIATION,
)
from experiment_analysis.data_models.ratio_metric_data import RatioMetricData
from experiment_analysis.stats.zstatistic import ZStatistic


class RatioMetricInference(AdditiveMetricInference):
    def __init__(
        self,
        data: RatioMetricData,
        num_bootstraps: int = 10000,
        num_randomizations: int = 10000,
    ) -> None:
        self.ratio_metric_data: RatioMetricData = data
        self.num_bootstraps = num_bootstraps
        self.num_randomizations = num_randomizations

        self._data = None
        self._treatment_effect = None
        self._metric = None
        self._assignment = None

        self._se_delta_method = None
        self._se_bootstrap = None

        self._ci_delta_method = None
        self._ci_bootstrap = None

    @property
    def data(self) -> NDArray[np.float64]:
        # a numpy array with columns, in order, metric_numerator, metric_denominator, assignment
        if self._data is None:
            self._data = np.hstack(
                (
                    self.metric.reshape((-1, 2)),
                    self.assignment.reshape((-1, 1)),
                )
            )  # type: ignore
        return self._data  # type: ignore

    @property
    def metric(self) -> NDArray[np.float64]:
        if self._metric is None:
            self._metric = np.array(self.ratio_metric_data.data[[METRIC_NUMERATOR, METRIC_DENOMINATOR]])  # type: ignore
        return self._metric  # type: ignore

    @property
    def assignment(self) -> NDArray[np.float64]:
        if self._assignment is None:
            variation = np.array(self.ratio_metric_data.data[VARIATION])
            self._assignment = np.where(variation == TREATMENT, 1, 0)  # type: ignore
        return self._assignment  # type: ignore

    @classmethod
    def estimate_treatment_effect(cls, data: NDArray[np.float64]) -> float:
        metric = data[:, :-1]
        assignment = data[:, -1]
        ratios: List[float] = []
        for variation in [0, 1]:
            variation_metric = metric[assignment == variation]
            numerator_mean = variation_metric[:, 0].mean()
            denominator_mean = variation_metric[:, 1].mean()
            variation_ratio = numerator_mean / denominator_mean
            ratios.append(variation_ratio)

        return ratios[1] - ratios[0]

    def get_p_value(self, method: str) -> float:
        if method == RANDOMIZATION:
            return self._get_p_value_randomization()
        elif method == BOOTSTRAP:
            return self._get_p_value_bootstrap()
        elif method == DELTA:
            return self._get_p_value_delta_method()
        elif method == FIELLER:
            raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def se_delta_method(self) -> float:
        if self._se_delta_method is None:
            # relies on CLT, assuming no outliers
            metric = self.data[:, :-1]
            assignment = self.data[:, -1]
            variances: List[float] = []
            for variation in [0, 1]:
                variation_metric = metric[assignment == variation]
                num_units = variation_metric.shape[0]

                numerator = variation_metric[:, 0]
                denominator = variation_metric[:, 1]

                mu_n, mu_d = numerator.mean(), denominator.mean()
                g_prime = np.array([1 / mu_d, -mu_n / (mu_d**2)]).reshape(
                    (1, 2)
                )

                covariance = np.cov(variation_metric.T) / num_units
                assert covariance.shape == (2, 2)

                variance = np.dot(g_prime, covariance)
                variance = np.dot(variance, g_prime.T)

                variances.append(variance)

            se = np.sqrt(variances[0] + variances[1])
            self._se_delta_method = se

        return self._se_delta_method  # type: ignore

    def _get_p_value_delta_method(self) -> float:
        return ZStatistic.get_p_value(
            self.treatment_effect, self.se_delta_method
        )

    def _get_p_value_fieller_method(self) -> float:
        pass

    def get_confidence_interval(
        self, level: float, method: str
    ) -> Tuple[float, float]:
        try:
            assert 0 <= level <= 1
        except AssertionError:
            raise ValueError("level should be between zero and 1")

        if method == BOOTSTRAP:
            se = self.se_bootstrap
        elif method == DELTA:
            se = self.se_delta_method
        else:
            raise NotImplementedError

        return ZStatistic.get_interval(self.treatment_effect, level, se)
