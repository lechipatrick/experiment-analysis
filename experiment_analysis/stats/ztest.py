import numpy as np
from scipy import stats


class ZTest:
    @classmethod
    def get_z_statistic(cls, estimate: float, std: float) -> float:
        return estimate / std

    @classmethod
    def get_p_value(cls, estimate: float, std: float) -> float:
        z_statistic = cls.get_z_statistic(estimate, std)
        return 2 * stats.norm.sf(np.abs(z_statistic), loc=0, scale=1)  # type: ignore
