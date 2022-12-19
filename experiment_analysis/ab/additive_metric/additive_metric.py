import numpy as np
import pandas as pd
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
from experiment_analysis.stats.ztest import ZTest


class AdditiveMetricInference:
    def __init__(self, data: AdditiveMetricData) -> None:
        self._data = data.data

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
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def metric(self) -> NDArray[np.float64]:
        if self._metric is None:
            self._metric = np.array(self.data[METRIC])  # type: ignore
        return self._metric  # type: ignore

    @property
    def assignment(self) -> NDArray[np.float64]:
        if self._assignment is None:
            variation = np.array(self.data[VARIATION])
            self._assignment = np.where(variation == TREATMENT, 1, 0)  # type: ignore
        return self._assignment  # type: ignore

    @classmethod
    def estimate_treatment_effect(
        cls, metric: NDArray[np.float64], assignment: NDArray[np.float64]
    ) -> float:
        control = metric[assignment == 0]
        treatment = metric[assignment == 1]
        return treatment.mean() - control.mean()  # type: ignore

    @property
    def treatment_effect(self) -> float:
        if not self._treatment_effect:
            self._treatment_effect = self.estimate_treatment_effect(
                self.metric, self.assignment
            )  # type: ignore
        return self._treatment_effect  # type: ignore

    def get_p_value(
        self, method: str = RANDOMIZATION, *args: int, **kwargs: int
    ) -> float:
        if method == RANDOMIZATION:
            return self._get_p_value_randomization(*args, **kwargs)
        elif method == BOOTSTRAP:
            return self._get_p_value_bootstrap(*args, **kwargs)
        elif method == ZTEST:
            return self._get_p_value_z_test(*args, **kwargs)
        else:
            raise NotImplementedError

    def _get_p_value_randomization(self, num_randomizations: int) -> float:
        randomized_estimates = (
            Randomization.get_simple_random_assignment_estimates(
                metric=self.metric,
                estimation_func=self.estimate_treatment_effect,  # type: ignore
                control_proportion=self.control_proportion,
                num_randomizations=num_randomizations,
            )
        )
        return Randomization.get_p_value(
            self.treatment_effect, randomized_estimates
        )

    def _get_p_value_z_test(self) -> float:
        # relies on CLT, assuming no outliers
        control = self.metric[self.assignment == 0]
        treatment = self.metric[self.assignment == 1]

        std = np.sqrt(
            control.var() / len(control) + treatment.var() / len(treatment)
        )
        return ZTest.get_p_value(self.treatment_effect, std)

    def _get_p_value_bootstrap(self, num_bootstraps: int) -> float:
        bootstrapped_estimates = Bootstrap.get_simple_bootstrapped_estimates(
            self.data,
            self.estimate_treatment_effect,  # type: ignore
            num_bootstraps,
        )
        return Bootstrap.get_p_value(
            self.treatment_effect, bootstrapped_estimates
        )

    def get_confidence_interval(
        self, level: float, method: str, *args: int, **kwargs: int
    ) -> float:
        pass
