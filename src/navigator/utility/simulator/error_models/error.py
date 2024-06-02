"""Defines the error models to be applied to the simulated epochs.

Classes:
    - BaseErrorModel: The base class for all error models.

"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm, poisson, rv_continuous, rv_discrete

from navigator.epoch import Epoch

from ....core.triangulate.itriangulate.algos.combinations.range_combinations import (
    L1_FREQ,
    L1_WAVELENGTH,
    L2_FREQ,
    L2_WAVELENGTH,
    SPEED_OF_LIGHT,
)

__all__ = [
    "BaseErrorModel",
    "MeasurementErrorModel",
    "IonoSphericalErrorModel",
    "TropoSphericalErrorModel",
    "CycleSlipsErrorModel",
    "ClockErrorModel",
    "ChainedErrorModel",
    "MultivariatePoisson",
    "IdentityErrorModel",
]


class BaseErrorModel(ABC):
    """The base class for all error models."""

    def __init__(self, error_model: str) -> None:
        """Initializes an instance of the BaseErrorModel class.

        Args:
            error_model (str): The error model to use.
        """
        self.error_model = error_model

    @abstractmethod
    def apply(self, epoch: Epoch) -> Epoch:
        """Applies the error model to the given epoch.

        Args:
            epoch (Epoch): The epoch to apply the error model to.

        Returns:
            Epoch: The epoch with the error model applied.
        """
        pass

    def __repr__(self) -> str:
        """Returns the string representation of the error model."""
        return f"{self.__class__.__name__}(error_model={self.error_model})"

    def __call__(self, epoch: Epoch) -> Epoch:
        """Applies the error model to the given epoch."""
        return self.apply(epoch)


class IdentityErrorModel(BaseErrorModel):
    """The error model for the identity error model."""

    def __init__(self) -> None:
        """Initializes an instance of the IdentityErrorModel class."""
        super().__init__(error_model="identity")

    def apply(self, epoch: Epoch) -> Epoch:
        """Applies the error model to the given epoch.

        Args:
            epoch (Epoch): The epoch to apply the error model to.

        Returns:
            Epoch: The epoch with the error model applied.
        """
        return epoch


class MeasurementErrorModel(BaseErrorModel):
    """The error model for the measurements error."""

    def __init__(self, rv_variable: rv_continuous) -> None:
        """Initializes an instance of the MeasurementErrorModel class.

        Args:
            rv_variable (rv_continuous): The random variable to use.
        """
        super().__init__(error_model="measurement")
        self.rv_variable = rv_variable

    def apply(self, epoch: Epoch) -> Epoch:
        """Applies the error model to the given epoch.

        Args:
            epoch (Epoch): The epoch to apply the error model to.

        Returns:
            Epoch: The epoch with the error model applied.
        """
        # Apply random errors on the observations on l1 frequency
        l1 = self.rv_variable.rvs(size=1).flatten()
        l2 = self.rv_variable.rvs(size=1).flatten()

        # Check if the error shape and epoch shape are the same
        if l1.shape[0] != epoch.obs_data.shape[0]:
            raise ValueError(
                f"Error shape {l1.shape} and epoch shape {epoch.obs_data.shape[0]} are not the same."
            )

        epoch.obs_data[epoch.L1_CODE_ON] += l1
        epoch.obs_data[epoch.L2_CODE_ON] += l2

        epoch.obs_data[epoch.L1_PHASE_ON] += l1
        epoch.obs_data[epoch.L2_PHASE_ON] += l2

        epoch.real_coord["L1_MEAS_ERROR"] = l1
        epoch.real_coord["L2_MEAS_ERROR"] = l2

        return epoch


class IonoSphericalErrorModel(BaseErrorModel):
    """The error model for the ionospheric spherical error."""

    def __init__(self, rv_variable: rv_continuous) -> None:
        """Initializes an instance of the IonoSphericalErrorModel class.

        Args:
            rv_variable (rv_continuous): The random variable to use.
        """
        super().__init__(error_model="iono_spherical")
        self.rv_variable = rv_variable

    def apply(self, epoch: Epoch) -> Epoch:
        """Applies the error model to the given epoch.

        Args:
            epoch (Epoch): The epoch to apply the error model to.

        Returns:
            Epoch: The epoch with the error model applied.
        """
        I = self.rv_variable.rvs(size=1).flatten()
        # Check if the error shape and epoch shape are the same
        if I.shape[0] != epoch.obs_data.shape[0]:
            raise ValueError(
                f"Error shape {I.shape} and epoch shape {epoch.obs_data.shape[0]} are not the same."
            )
        I_2 = I * (L2_FREQ / L1_FREQ) ** 2

        # Apply ionospheric spherical error
        epoch.obs_data[epoch.L1_CODE_ON] += I
        epoch.obs_data[epoch.L2_CODE_ON] += I_2

        # Apply ionospheric spherical error on phase measurements
        epoch.obs_data[epoch.L1_PHASE_ON] -= I
        epoch.obs_data[epoch.L2_PHASE_ON] -= I_2

        epoch.real_coord["IONO_ERROR"] = I

        return epoch


class TropoSphericalErrorModel(BaseErrorModel):
    """The error model for the tropospheric spherical error."""

    def __init__(self, rv_variable: rv_continuous) -> None:
        """Initializes an instance of the TropoSphericalErrorModel class.

        Args:
            rv_variable (rv_continuous): The random variable to use.
        """
        super().__init__(error_model="tropo_spherical")
        self.rv_variable = rv_variable

    def apply(self, epoch: Epoch) -> Epoch:
        """Applies the error model to the given epoch.

        Args:
            epoch (Epoch): The epoch to apply the error model to.

        Returns:
            Epoch: The epoch with the error model applied.
        """
        T = self.rv_variable.rvs(size=1).flatten()
        # Check if the error shape and epoch shape are the same
        if T.shape[0] != epoch.obs_data.shape[0]:
            raise ValueError(
                f"Error shape {T.shape} and epoch shape {epoch.obs_data.shape[0]} are not the same."
            )
        # Apply tropospheric spherical error
        epoch.obs_data[epoch.L1_CODE_ON] += T
        epoch.obs_data[epoch.L2_CODE_ON] += T

        # Apply tropospheric spherical error on phase measurements
        epoch.obs_data[epoch.L1_PHASE_ON] += T
        epoch.obs_data[epoch.L2_PHASE_ON] += T

        epoch.real_coord["TROPO_ERROR"] = T

        return epoch


class CycleSlipsErrorModel(BaseErrorModel):
    """The error model for the cycle slips error."""

    def __init__(self, rv_variable: rv_discrete, freq: int = 100) -> None:
        """Initializes an instance of the CycleSlipsErrorModel class.

        Args:
            rv_variable (rv_discrete): The random variable to use. (Must be multinomial distribution to account for multiple satellites at once)
            freq (int): The frequency of the cycle slips.
        """
        super().__init__(error_model="cycle_slips")
        self.rv_variable = rv_variable
        self.freq = freq
        self.counter = 0

        # Initialize the cycle slips
        self.slips = self.rv_variable.rvs(size=1)

    def apply(self, epoch: Epoch) -> Epoch:
        """Applies the error model to the given epoch.

        Args:
            epoch (Epoch): The epoch to apply the error model to.

        Returns:
            Epoch: The epoch with the error model applied.
        """
        # Check if the counter is equal to the frequency
        if self.counter == self.freq:
            # Reset the counter
            self.counter = 0

            # Generate new cycle slips
            self.slips = self.rv_variable.rvs(size=1).flatten()

            # Check if the error shape and epoch shape are the same
            if self.slips.shape[0] != epoch.obs_data.shape[0]:
                raise ValueError(
                    f"Error shape {self.slips.shape} and epoch shape {epoch.obs_data.shape[0]} are not the same."
                )

        # Apply cycle slips
        epoch.obs_data[epoch.L1_PHASE_ON] += L1_WAVELENGTH * self.slips
        epoch.obs_data[epoch.L2_PHASE_ON] += L2_WAVELENGTH * self.slips

        # Increment the counter
        self.counter += 1

        # Store the cycle slips
        epoch.real_coord["CYCLE_SLIPS"] = self.slips

        return epoch


class ClockErrorModel(BaseErrorModel):
    """The error model for the clock error."""

    def __init__(self, rv_variable: rv_continuous) -> None:
        """Initializes an instance of the ClockErrorModel class.

        Args:
            rv_variable (rv_continuous): The random variable to use.
        """
        super().__init__(error_model="clock")
        self.rv_variable = rv_variable

    def apply(self, epoch: Epoch) -> Epoch:
        """Applies the error model to the given epoch.

        Args:
            epoch (Epoch): The epoch to apply the error model to.

        Returns:
            Epoch: The epoch with the error model applied.
        """
        dt = self.rv_variable.rvs(size=1).flatten()

        # Check if the error shape and epoch shape are the same
        if dt.shape[0] != epoch.obs_data.shape[0]:
            raise ValueError(
                f"Error shape {dt.shape} and epoch shape {epoch.obs_data.shape[0]} are not the same."
            )

        # Apply clock error
        epoch.obs_data[epoch.L1_CODE_ON] += dt * SPEED_OF_LIGHT
        epoch.obs_data[epoch.L2_CODE_ON] += dt * SPEED_OF_LIGHT

        # Apply clock error on phase measurements
        epoch.obs_data[epoch.L1_PHASE_ON] += dt * SPEED_OF_LIGHT
        epoch.obs_data[epoch.L2_PHASE_ON] += dt * SPEED_OF_LIGHT

        epoch.real_coord["CLOCK_ERROR"] = dt

        return epoch


class ChainedErrorModel(BaseErrorModel):
    """The error model for the chained error model."""

    def __init__(self, error_models: list) -> None:
        """Initializes an instance of the ChainedErrorModel class.

        Args:
            error_models (list): The list of error models to chain.
        """
        super().__init__(error_model="chained")
        self.error_models = error_models

    def apply(self, epoch: Epoch) -> Epoch:
        """Applies the error model to the given epoch.

        Args:
            epoch (Epoch): The epoch to apply the error model to.

        Returns:
            Epoch: The epoch with the error model applied.
        """
        for error_model in self.error_models:
            epoch = error_model.apply(epoch)

        return epoch


class MultivariatePoisson(rv_continuous):
    """Multivariate Poisson distribution.

    This class represents a multivariate Poisson distribution, which is a generalization of the Poisson distribution
    to multiple dimensions. It generates samples from a multivariate Poisson distribution using the specified
    lambdas and correlation matrix.

    Parameters:
    - lambdas (array-like): The mean parameters for each dimension of the distribution.
    - corr_matrix (array-like): The correlation matrix specifying the correlation between dimensions.

    Example usage:
    >>> lambdas = [2, 3, 4]
    >>> corr_matrix = [[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]]
    >>> dist = MultivariatePoisson(lambdas, corr_matrix)
    >>> samples = dist.rvs(size=100)
    """

    def __init__(
        self,
        lambdas: List[float],
        corr_matrix: List[List[float]],
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the MultivariatePoisson distribution.

        Args:
            lambdas (List[float]): The mean parameters for each dimension of the distribution.
            corr_matrix (List[List[float]]): The correlation matrix specifying the correlation between dimensions.
            seed (Optional[int]): Random seed for reproducibility.
        """
        super().__init__()
        self.lambdas = lambdas
        self.corr_matrix = corr_matrix
        self.seed = seed

    def _rvs(
        self,
        size: Optional[Union[int, Tuple[int]]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> np.ndarray:
        """Generate random samples from the multivariate Poisson distribution.

        This method generates random samples from the multivariate Poisson distribution using the specified
        lambdas and correlation matrix.

        Args:
            size (Optional[Union[int, Tuple[int]]]): The shape of the output samples. If None, a single sample is generated.
            random_state (Optional[Union[int, np.random.RandomState]]): Random seed or random state object.

        Returns:
            np.ndarray: Random samples from the multivariate Poisson distribution.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Step 1: Generate samples from a multivariate normal distribution
        mean = np.zeros(len(self.lambdas))
        mvn_samples = np.random.multivariate_normal(mean, self.corr_matrix, size)

        # Step 2: Transform the normal samples to uniform samples using the CDF
        uniform_samples = norm.cdf(mvn_samples)

        # Step 3: Transform the uniform samples to Poisson samples using the inverse CDF
        poisson_samples = np.zeros_like(uniform_samples)
        for i, lam in enumerate(self.lambdas):
            poisson_samples[:, i] = poisson.ppf(uniform_samples[:, i], lam)

        return poisson_samples.flatten()
