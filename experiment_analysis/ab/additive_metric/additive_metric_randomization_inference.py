import numpy as np
import pandas as pd
from numpy.typing import NDArray

from experiment_analysis.base.base import (
    estimate_treatment_effect_additive_metric,
    get_control_proportion,
    get_p_value_randomized_inference,
    randomize_assignment,
    validate_data_columns,
    validate_data_type,
    validate_num_draws,
    validate_variation,
)
from experiment_analysis.constants import METRIC, VARIATION


class AdditiveMetricRandomizationInference:
    def __init__(self, *, data: pd.DataFrame, num_draws: int = 10000) -> None:
        self.data = data
        self.num_draws = num_draws
        self._control_proportion = None
        self._treatment_effect = None

        self._validate()

    def _validate(self) -> None:
        validate_data_type(self.data)
        validate_data_columns(self.data, columns=[METRIC, VARIATION])
        validate_num_draws(self.num_draws)
        validate_variation(self.data)

    @property
    def treatment_effect(self) -> float:
        if not self._treatment_effect:
            self._treatment_effect = estimate_treatment_effect_additive_metric(
                self.data
            )
        return self._treatment_effect  # type: ignore

    @property
    def control_proportion(self) -> float:
        if not self._control_proportion:
            self._control_proportion = get_control_proportion(self.data)
        return self._control_proportion  # type: ignore

    def draw_treatment_effects(self) -> NDArray[np.float64]:
        drawn_treatment_effects = np.zeros((self.num_draws,))

        for i in range(self.num_draws):
            drawn_data = randomize_assignment(
                self.data, self.control_proportion
            )
            treatment_effect = estimate_treatment_effect_additive_metric(
                drawn_data
            )
            drawn_treatment_effects[i] = treatment_effect

        return drawn_treatment_effects

    def get_p_value(self) -> float:
        drawn_treatment_effects = self.draw_treatment_effects()
        observe_treatment_effect = estimate_treatment_effect_additive_metric(
            self.data
        )
        return get_p_value_randomized_inference(
            observe_treatment_effect, drawn_treatment_effects
        )
