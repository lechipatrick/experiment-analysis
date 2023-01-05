from typing import Tuple

import numpy as np
from scipy import stats


class ZStatistic:
    @classmethod
    def get_z_statistic(cls, estimate: float, std: float) -> float:
        return estimate / std

    @classmethod
    def get_p_value(cls, estimate: float, std: float) -> float:
        z_statistic = cls.get_z_statistic(estimate, std)
        return 2 * stats.norm.sf(np.abs(z_statistic), loc=0, scale=1)  # type: ignore

    @classmethod
    def get_critical_value(cls, level: float) -> float:
        try:
            assert 0 <= level <= 1
        except AssertionError:
            raise ValueError("level should be between zero and 1")
        return stats.norm.ppf((1 + level) / 2, loc=0, scale=1)  # type: ignore

    @classmethod
    def get_interval(
        cls, mean: float, level: float, se: float
    ) -> Tuple[float, float]:
        threshold = cls.get_critical_value(level)
        return mean - se * threshold, mean + se * threshold
