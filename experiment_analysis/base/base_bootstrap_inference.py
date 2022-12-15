import pandas as pd
from scipy import stats

from experiment_analysis.base.base_simulation_inference import (
    BaseSimulationInference,
)


class BaseBootstrapInference(BaseSimulationInference):
    def __init__(self, *, data: pd.DataFrame, num_draws: int) -> None:
        super().__init__(data=data, num_draws=num_draws)

    def draw_data(self) -> pd.DataFrame:
        return self.data.sample(frac=1, replace=True, ignore_index=True)

    def estimate_treatment_effect(self, data: pd.DataFrame) -> float:
        raise NotImplementedError

    def p_value(self) -> float:
        # assumes that bootstrapped estimates follow a normal distribution
        drawn_treatment_effects = self.draw_treatment_effects()
        se = drawn_treatment_effects.std()
        z_statistic = self.treatment_effect / se
        p_value = 2 * (stats.norm.sf(z_statistic, loc=0, scale=1))
        return p_value  # type: ignore
