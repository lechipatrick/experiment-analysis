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
            variation_ratio = (
                variation_metric[:, 0].mean() / variation_metric[:, 1].mean()
            )
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
            raise NotImplementedError
        elif method == FIELLER:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _get_p_value_randomization(self, num_randomizations: int) -> float:
        randomization_estimator = Randomization(self.data)
        randomization_estimates = (
            randomization_estimator.get_simple_randomized_assignment_estimates(
                estimation_func=self.estimate_treatment_effect,
                num_randomizations=num_randomizations,
            )
        )
        return randomization_estimator.get_p_value(
            self.treatment_effect, randomization_estimates
        )

    def _get_p_value_delta_method(self) -> float:
        # relies on CLT, assuming no outliers
        control = self.metric[self.assignment == 0]
        treatment = self.metric[self.assignment == 1]

        se = np.sqrt(
            control.var() / len(control) + treatment.var() / len(treatment)
        )
        return ZStatistic.get_p_value(self.treatment_effect, se)

    def _get_p_value_fieller_method(self) -> float:
        pass

    def _get_p_value_bootstrap(self, num_bootstraps: int) -> float:
        bootstrapper = Bootstrap(self.data)
        bootstrap_estimates = bootstrapper.get_bootstrap_estimates(
            self.estimate_treatment_effect,
            num_bootstraps,
        )
        return bootstrapper.get_p_value(
            self.treatment_effect, bootstrap_estimates
        )

    def get_confidence_interval(
        self, level: float, method: str, *args: int, **kwargs: int
    ) -> float:
        pass
