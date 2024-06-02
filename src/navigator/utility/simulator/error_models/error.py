"""Defines the error models to be applied to the simulated epochs.


Classes:
    - BaseErrorModel: The base class for all error models.

"""

from abc import ABC, abstractmethod
from navigator.epoch import Epoch
from scipy.stats import rv_continuous, rv_discrete, multivariate_normal, poisson
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
        return f"{self.__class__.__name__}(error_model={self.error_model})"

    def __call__(self, epoch: Epoch) -> Epoch:
        return self.apply(epoch)


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
        l1 = self.rv_variable.rvs(size=1)
        l2 = self.rv_variable.rvs(size=1)

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
        I = self.rv_variable.rvs(size=1)
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
        T = self.rv_variable.rvs(size=1)

        

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
            self.slips = self.rv_variable.rvs(size=1)

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
        dt = self.rv_variable.rvs(size=1)

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



# Multivarite possian distribution
poisson_rv = 


def gaussian_error_model(num_sv: int) -> ChainedErrorModel:
    """Creates a chained error model with the following error models:
    - MeasurementErrorModel
    - IonoSphericalErrorModel
    - TropoSphericalErrorModel
    - CycleSlipsErrorModel
    - ClockErrorModel

    Args:
        num_sv (int): The number of satellites to simulate.

    Returns:
        ChainedErrorModel: The chained error model.
    """
    # Multivariate Gaussian Model
    gaussian_rv = s
    return ChainedErrorModel(
        [
            MeasurementErrorModel(rv_variable=rv_continuous(name="gaussian", a=0, b=1)),
            IonoSphericalErrorModel(rv_variable=rv_continuous(name="gaussian", a=0, b=1)),
            TropoSphericalErrorModel(rv_variable=rv_continuous(name="gaussian", a=0, b=1)),
            CycleSlipsErrorModel(rv_variable=rv_discrete(name="multinomial", values=(0, 1), p=(0.99, 0.01)), freq=100),
            ClockErrorModel(rv_variable=rv_continuous(name="gaussian", a=0, b=1)),
        ]
    )