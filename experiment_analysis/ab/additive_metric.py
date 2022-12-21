import numpy as np
from numpy.typing import NDArray

from experiment_analysis.constants import (
    BOOTSTRAP,
    METRIC,
    RANDOMIZATION,
    TREATMENT,
    VARIATION,
    ZTEST,
)
from experiment_analysis.data_models.additive_metric_data import (
    AdditiveMetricData,
)
from experiment_analysis.stats.bootstrap import Bootstrap
from experiment_analysis.stats.randomization import Randomization
from experiment_analysis.stats.zstatistic import ZStatistic


class AdditiveMetricInference:
    def __init__(self, data: AdditiveMetricData) -> None:
        self.additive_metric_data: AdditiveMetricData = data

        self._data = None
        self._treatment_effect = None
        self._metric = None
        self._assignment = None
        self._control_proportion = None

    @property
    def control_proportion(self) -> float:
        if self._control_proportion is None:
            self._control_proportion = 1 - self.assignment.mean()
        return self._control_proportion  # type: ignore

    @property
    def data(self) -> NDArray[np.float64]:
        if self._data is None:
            self._data = np.hstack(
                (
                    self.metric.reshape((-1, 1)),
                    self.assignment.reshape((-1, 1)),
                )
            )  # type: ignore
        return self._data  # type: ignore

    @property
    def metric(self) -> NDArray[np.float64]:
        if self._metric is None:
            self._metric = np.array(self.additive_metric_data.data[METRIC])  # type: ignore
        return self._metric  # type: ignore

    @property
    def assignment(self) -> NDArray[np.float64]:
        if self._assignment is None:
            variation = np.array(self.additive_metric_data.data[VARIATION])
            self._assignment = np.where(variation == TREATMENT, 1, 0)  # type: ignore
        return self._assignment  # type: ignore

    @classmethod
    def estimate_treatment_effect(cls, data: NDArray[np.float64]) -> float:
        metric = data[:, :-1]
        assignment = data[:, -1]
        control = metric[assignment == 0]
        treatment = metric[assignment == 1]
        return treatment.mean() - control.mean()  # type: ignore

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
        elif method == ZTEST:
            return self._get_p_value_z_test(*args, **kwargs)
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

    def _get_p_value_z_test(self) -> float:
        # relies on CLT, assuming no outliers
        control = self.metric[self.assignment == 0]
        treatment = self.metric[self.assignment == 1]

        se = np.sqrt(
            control.var() / len(control) + treatment.var() / len(treatment)
        )
        return ZStatistic.get_p_value(self.treatment_effect, se)

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
