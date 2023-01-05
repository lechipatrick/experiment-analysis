from typing import List

import numpy as np
from numpy.typing import NDArray

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
from experiment_analysis.stats.bootstrap import Bootstrap
from experiment_analysis.stats.randomization import Randomization
from experiment_analysis.stats.zstatistic import ZStatistic


class RatioMetricInference:
    def __init__(self, data: RatioMetricData) -> None:
        self.ratio_metric_data: RatioMetricData = data

        self._data = None
        self._treatment_effect = None
        self._metric = None
        self._assignment = None

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

    @property
    def treatment_effect(self) -> float:
        if not self._treatment_effect:
            self._treatment_effect = self.estimate_treatment_effect(
                self.data
            )  # type: ignore
        return self._treatment_effect  # type: ignore

    def get_p_value(self, method: str, *args: int, **kwargs: int) -> float:
        if method == RANDOMIZATION:
            return self._get_p_value_randomization(*args, **kwargs)
        elif method == BOOTSTRAP:
            return self._get_p_value_bootstrap(*args, **kwargs)
        elif method == DELTA:
            return self._get_p_value_delta_method(*args, **kwargs)
        elif method == FIELLER:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _get_p_value_randomization(self, num_randomizations: int) -> float:
        randomization_estimator = Randomization(
            self.data, self.estimate_treatment_effect, num_randomizations
        )
        randomization_estimates = (
            randomization_estimator.get_randomized_assignment_estimates()
        )
        return randomization_estimator.get_p_value(
            self.treatment_effect, randomization_estimates
        )

    def _get_p_value_delta_method(self) -> float:
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
            g_prime = np.array([1 / mu_d, -mu_n / (mu_d**2)]).reshape((1, 2))

            covariance = np.cov(variation_metric.T) / num_units
            assert covariance.shape == (2, 2)

            variance = np.dot(g_prime, covariance)
            variance = np.dot(variance, g_prime.T)

            variances.append(variance)

        se = np.sqrt(variances[0] + variances[1])

        return ZStatistic.get_p_value(self.treatment_effect, se)

    def _get_p_value_fieller_method(self) -> float:
        pass

    def _get_p_value_bootstrap(self, num_bootstraps: int) -> float:
        bootstrapper = Bootstrap(
            self.data, self.estimate_treatment_effect, num_bootstraps
        )
        bootstrap_estimates = bootstrapper.get_bootstrap_estimates()
        return bootstrapper.get_p_value(
            self.treatment_effect, bootstrap_estimates
        )

    def get_confidence_interval(
        self, level: float, method: str, *args: int, **kwargs: int
    ) -> float:
        pass


# import pandas as pd
#
# def generate_test_data(
#     num_units: int,
#     treatment_effect: float,
#     cov: NDArray[np.float64] = np.array([[1, 0], [0, 1]]),
#     control_proportion: float = 0.5,
# ) -> RatioMetricData:
#     # TODO: figure out how to do this as a pytest.fixture
#
#     control_num_units = int(num_units * control_proportion)
#     treatment_num_units = int(num_units * (1 - control_proportion))
#
#     control_mean = np.array([1, 1])
#     treatment_mean = np.array([1 + treatment_effect, 1])
#     variation_control = ["control" for _ in range(control_num_units)]
#     variation_treatment = ["treatment" for _ in range(treatment_num_units)]
#
#     (
#         metric_control_numerator,
#         metric_control_denominator,
#     ) = np.random.multivariate_normal(control_mean, cov, control_num_units).T
#     (
#         metric_treatment_numerator,
#         metric_treatment_denominator,
#     ) = np.random.multivariate_normal(
#         treatment_mean, cov, treatment_num_units
#     ).T
#
#     data = {
#         METRIC_NUMERATOR: np.hstack(
#             (metric_control_numerator, metric_treatment_numerator)
#         ),
#         METRIC_DENOMINATOR: np.hstack(
#             (metric_control_denominator, metric_treatment_denominator)
#         ),
#         VARIATION: variation_control + variation_treatment,
#     }
#     df = pd.DataFrame.from_dict(data)
#     return RatioMetricData(data=df)
#
# num_sims = 1000
# num_units = 1000
#
# from tqdm import tqdm
# p_values = np.zeros(num_sims)
# for i in tqdm(range(num_sims)):
#     test_data = generate_test_data(num_units=num_units, treatment_effect=0)
#     inference = RatioMetricInference(test_data)
#     p_value = inference.get_p_value("delta")
#     p_values[i] = p_value
# print(p_values.min(), p_values.max(), p_values.mean())
