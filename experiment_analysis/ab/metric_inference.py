from typing import Tuple
from abc import abstractmethod, ABC
import numpy as np
from numpy.typing import NDArray

from experiment_analysis.stats.bootstrap import Bootstrap
from experiment_analysis.stats.randomization import Randomization
from experiment_analysis.stats.zstatistic import ZStatistic
from experiment_analysis.utils.log import get_logger

logger = get_logger(__name__)


class MetricInference(ABC):

    @property
    @abstractmethod
    def data(self) -> NDArray[np.float64]:
        # metric and assignment status for each unit (row)
        pass

    @property
    @abstractmethod
    def metric(self) -> NDArray[np.float64]:
        # values for the metric for each unit (row)
        pass

    @property
    def assignment(self) -> NDArray[np.float64]:
        # assignment status for each unit (row)
        pass

    @classmethod
    def estimate_treatment_effect(cls, data: NDArray[np.float64]) -> float:
        pass

    @property
    def treatment_effect(self) -> float:
        if not self._treatment_effect:
            self._treatment_effect = self.estimate_treatment_effect(
                self.data
            )  # type: ignore
        return self._treatment_effect  # type: ignore

    @abstractmethod
    def get_p_value(self, method: str) -> float:
        pass

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

    def get_p_value_bootstrap(self) -> float:
        return ZStatistic.get_p_value(self.treatment_effect, self.se_bootstrap)

    @abstractmethod
    def get_confidence_interval(
        self, level: float, method: str
    ) -> Tuple[float, float]:
        pass
