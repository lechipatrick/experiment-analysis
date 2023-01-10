from typing import Tuple

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
from experiment_analysis.stats.zstatistic import ZStatistic
from experiment_analysis.utils.log import get_logger
from experiment_analysis.ab.metric_inference import MetricInference

logger = get_logger(__name__)


class AdditiveMetricInference(MetricInference):
    def __init__(
        self,
        data: AdditiveMetricData,
        num_bootstraps: int = 10000,
        num_randomizations: int = 10000,
    ) -> None:
        self.metric_data: AdditiveMetricData = data
        self.num_bootstraps = num_bootstraps
        self.num_randomizations = num_randomizations

        self._data = None
        self._treatment_effect = None
        self._metric = None
        self._assignment = None

        self._se_z_test = None
        self._se_bootstrap = None

        self._ci_z_test = None
        self._ci_bootstrap = None

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
            self._metric = np.array(self.metric_data.data[METRIC])  # type: ignore
        return self._metric  # type: ignore

    @property
    def assignment(self) -> NDArray[np.float64]:
        if self._assignment is None:
            variation = np.array(self.metric_data.data[VARIATION])
            self._assignment = np.where(variation == TREATMENT, 1, 0)  # type: ignore
        return self._assignment  # type: ignore

    @classmethod
    def estimate_treatment_effect(cls, data: NDArray[np.float64]) -> float:
        metric = data[:, :-1]
        assignment = data[:, -1]
        control = metric[assignment == 0]
        treatment = metric[assignment == 1]
        return treatment.mean() - control.mean()  # type: ignore

    def get_p_value(self, method: str) -> float:
        if method == RANDOMIZATION:
            return self._get_p_value_randomization()
        elif method == BOOTSTRAP:
            return self._get_p_value_bootstrap()
        elif method == ZTEST:
            return self._get_p_value_z_test()
        else:
            raise NotImplementedError

    def _get_p_value_z_test(self) -> float:
        return ZStatistic.get_p_value(self.treatment_effect, self.se_z_test)

    @property
    def se_z_test(self) -> float:
        if self._se_z_test is None:

            # relies on CLT, assuming no outliers
            control = self.metric[self.assignment == 0]
            treatment = self.metric[self.assignment == 1]

            se = np.sqrt(
                control.var() / len(control) + treatment.var() / len(treatment)
            )
            self._se_z_test = se
        return self._se_z_test  # type: ignore

    def get_confidence_interval(
        self, level: float, method: str
    ) -> Tuple[float, float]:
        try:
            assert 0 <= level <= 1
        except AssertionError:
            raise ValueError("level should be between zero and 1")

        if method == BOOTSTRAP:
            se = self.se_bootstrap
        elif method == ZTEST:
            se = self.se_z_test
        else:
            raise NotImplementedError

        return ZStatistic.get_interval(self.treatment_effect, level, se)