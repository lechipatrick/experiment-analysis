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
from experiment_analysis.stats.bootstrap import Bootstrap
from experiment_analysis.stats.randomization import Randomization
from experiment_analysis.stats.zstatistic import ZStatistic
from experiment_analysis.utils.log import get_logger

logger = get_logger(__name__)


class AdditiveMetricInference:
    def __init__(
        self,
        data: AdditiveMetricData,
    ) -> None:
        self.metric_data: AdditiveMetricData = data
        self._num_randomizations = None
        self._num_bootstraps = None

        self._data = None
        self._treatment_effect = None
        self._metric = None
        self._assignment = None

        self._se_z_test = None
        self._se_bootstrap = None

        self._ci_z_test = None
        self._ci_bootstrap = None

    def set_num_randomizations(self, num_randomizations: int) -> None:
        self._num_randomizations = num_randomizations

    def set_num_bootstraps(self, num_bootstraps: int) -> None:
        self._num_bootstraps = num_bootstraps

    @property
    def num_randomizations(self) -> int:
        if self._num_randomizations is None:
            raise ValueError("num_randomizations is not specified. use set_num_randomizations()")
        else:
            return self._num_randomizations

    @property
    def num_bootstraps(self) -> int:
        if self._num_bootstraps is None:
            raise ValueError("num_randomizations is not specified. use set_num_randomizations()")
        else:
            return self._num_bootstraps

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

    @property
    def treatment_effect(self) -> float:
        if not self._treatment_effect:
            self._treatment_effect = self.estimate_treatment_effect(
                self.data
            )  # type: ignore
        return self._treatment_effect  # type: ignore

    @classmethod
    def estimate_treatment_effect(cls, data: NDArray[np.float64]) -> float:
        metric = data[:, :-1]
        assignment = data[:, -1]
        control = metric[assignment == 0]
        treatment = metric[assignment == 1]
        return treatment.mean() - control.mean()  # type: ignore

    def get_p_value_bootstrap(self) -> float:
        return ZStatistic.get_p_value(self.treatment_effect, self.se_bootstrap)

    @property
    def se_bootstrap(self) -> float:
        if self._se_bootstrap is None:
            logger.info(f"using number of bootstraps {self.num_bootstraps}")
            bootstrapper = Bootstrap(
                self.data, self.estimate_treatment_effect, self.num_bootstraps
            )
            bootstrap_estimates = bootstrapper.get_bootstrap_estimates()
            se = bootstrap_estimates.std()
            self._se_bootstrap = se
        return self._se_bootstrap  # type: ignore

    def get_p_value_randomization(self) -> float:
        logger.info(
            f"using number of randomizations {self.num_randomizations}"
        )
        randomization_estimator = Randomization(
            self.data, self.estimate_treatment_effect, self.num_randomizations
        )
        randomization_estimates = (
            randomization_estimator.get_randomized_assignment_estimates()
        )
        return randomization_estimator.get_p_value(
            self.treatment_effect, randomization_estimates
        )

    def get_p_value(self, method: str) -> float:
        if method == RANDOMIZATION:
            return self.get_p_value_randomization()
        elif method == BOOTSTRAP:
            return self.get_p_value_bootstrap()
        elif method == ZTEST:
            return self.get_p_value_z_test()
        else:
            raise NotImplementedError

    def get_p_value_z_test(self) -> float:
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
