import pandas as pd

from experiment_analysis.constants import CONTROL, METRIC, TREATMENT, VARIATION


class BaseAdditiveMetric:
    @classmethod
    def estimate_treatment_effect(cls, data: pd.DataFrame) -> float:
        control_mean = data[data[VARIATION] == CONTROL][METRIC].mean()
        treatment_mean = data[data[VARIATION] == TREATMENT][METRIC].mean()
        return treatment_mean - control_mean  # type: ignore
