import numpy as np
import pandas as pd
from numpy.typing import NDArray

from experiment_analysis.base.base import (
    bootstrap,
    estimate_treatment_effect_additive_metric,
    get_p_value_bootstrap,
)


class AdditiveMetricBootstrapInference:
    def __init__(self, *, data: pd.DataFrame, num_draws: int = 10000) -> None:
        self.data = data
        self.num_draws = num_draws
        self._treatment_effect = None

    @property
    def treatment_effect(self) -> float:
        if not self._treatment_effect:
            self._treatment_effect = estimate_treatment_effect_additive_metric(
                self.data
            )
        return self._treatment_effect  # type: ignore

    def draw_treatment_effects(self) -> NDArray[np.float64]:
        drawn_treatment_effects = np.zeros((self.num_draws,))

        for i in range(self.num_draws):
            drawn_data = bootstrap(self.data)
            treatment_effect = estimate_treatment_effect_additive_metric(
                drawn_data
            )
            drawn_treatment_effects[i] = treatment_effect

        return drawn_treatment_effects

    def get_p_value(self) -> float:
        drawn_treatment_effects = self.draw_treatment_effects()
        observed_treatment_effect = self.treatment_effect
        return get_p_value_bootstrap(observed_treatment_effect, drawn_treatment_effects)
