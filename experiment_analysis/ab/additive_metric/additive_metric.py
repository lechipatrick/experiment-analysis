import pandas as pd
from experiment_analysis.constants import (
    BOOTSTRAP,
    CONTROL,
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


class AdditiveMetricInference:
    def __init__(self, data: AdditiveMetricData) -> None:
        self.data = data.data

        self._treatment_effect = None

    @classmethod
    def get_control_proportion(cls, data: pd.DataFrame) -> float:
        variation_counts = data[VARIATION].value_counts().to_dict()
        control_proportion = variation_counts[CONTROL] / (
            variation_counts[CONTROL] + variation_counts[TREATMENT]
        )
        return control_proportion  # type: ignore

    @classmethod
    def estimate_treatment_effect(
        cls, data: pd.DataFrame
    ) -> float:
        control_mean = data[data[VARIATION] == CONTROL][METRIC].mean()
        treatment_mean = data[data[VARIATION] == TREATMENT][METRIC].mean()

        return treatment_mean - control_mean  # type: ignore

    @property
    def treatment_effect(self) -> float:
        if not self._treatment_effect:
            self._treatment_effect = (
                self.estimate_treatment_effect(self.data)
            )
        return self._treatment_effect  # type: ignore

    def get_p_value(self, method=RANDOMIZATION, *args, **kwargs):
        if method == RANDOMIZATION:
            return self._get_p_value_randomization(*args, **kwargs)
        elif method == BOOTSTRAP:
            return self._get_p_value_bootstrap(*args, **kwargs)
        elif method == ZTEST:
            raise NotImplementedError
            # return self._get_p_value_z_test(*args, **kwargs)

    def _get_p_value_bootstrap(self, num_bootstraps: int) -> float:
        bootstrapped_estimates = Bootstrap.get_simple_bootstrapped_estimates(
            self.data,
            self.estimate_treatment_effect,
            num_bootstraps,
        )
        return Bootstrap.get_p_value(
            self.treatment_effect, bootstrapped_estimates
        )

    def _get_p_value_randomization(self, num_randomizations: int) -> float:
        randomized_estimates = (
            Randomization.get_simple_randomized_assignment_estimates(
                self.data,
                self.estimate_treatment_effect,
                self.get_control_proportion(self.data),
                num_randomizations,
            )
        )
        return Randomization.get_p_value(
            self.treatment_effect, randomized_estimates
        )
