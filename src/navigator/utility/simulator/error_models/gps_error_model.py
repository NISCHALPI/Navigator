r"""Receiver Simulator Module.

This module contains the ReceiverSimulator class, which simulates the position and measurements of a GPS receiver.

Classes:
    ReceiverSimulator:
        Simulates the behavior of a GPS receiver, including position updates and pseudorange measurements.

Usage Example:
    # Initialize the receiver simulator
    receiver = ReceiverSimulator(initial_coords=(1115077.69025948, -4843958.49112974, 3983260.99261736),
                                  velocity=(5, 5, 5),
                                  mean_psuedorange_error=10,
                                  std_psuedorange_error=5,
                                  mean_satellite_error=1,
                                  std_satellite_error=1)

    # Set simulation time
    simulation_time = 3600  # 1 hour in seconds

    # Simulate receiver measurements at different times
    for t in range(simulation_time + 1):
        true_coords, satellite_measurements = receiver.get_measurements(t)

        # Print results for each measurement
        print(f"\nTime: {t} seconds")
        print("True Coordinates of Receiver:", true_coords)
        print("Satellite Measurements:")
        print(satellite_measurements)

Attributes:
    constellaion (GPSConstellation):
        GPS constellation instance used by the ReceiverSimulator.

Methods:
    __init__(self, initial_coords, velocity, mean_psuedorange_error, std_psuedorange_error, mean_satellite_error, std_satellite_error):
        Initializes the ReceiverSimulator object with specified parameters.
    update_position(self, t):
        Updates the position of the receiver based on its velocity.
    get_measurements(self, t):
        Simulates receiver measurements at a given time, including true coordinates and pseudorange measurements.

Note:
    The parameters for initialization (initial_coords, velocity, mean_psuedorange_error, etc.) should be adjusted based on the specific scenario you want to simulate.
"""

import numpy as np

__all__ = ["TimeVaryingErrorModel", "GPSErrorModel"]


class TimeVaryingErrorModel:
    """Simulates the behavior of a GPS receiver clock naively.

    The clock error is modelled as a simple sinusoidal function of time with a mean and amplitude with random noise
    sampled from a random walk process.


    Args:
        bias (float): The mean of the clock error.
        drift (float): The amplitude of the clock error.
        random_noise (float): The standard deviation of the random walk process.
    """

    def __init__(
        self,
        bias: float = 1e-5,
        drift: float = 3e-6,
        random_noise: float = 1e-6,
    ) -> None:
        """Initialize the clock model.

        Args:
            bias (float): The mean of the clock error.
            drift (float): The amplitude of the clock error.
            random_noise (float): The standard deviation of the random walk process.

        Returns:
            None
        """
        if bias < 0.0:
            raise ValueError("Clock bias must be non-negative.")
        if drift < 0.0:
            raise ValueError("Clock drift must be non-negative.")

        self.bias = bias
        self.drift = drift

        if random_noise < 0.0:
            raise ValueError("Random noise must be non-negative.")

        self.random_noise = random_noise

    def get_error(self, t: float) -> float:
        """Get the error at a given time.

        Args:
            t (float): Time elapsed from the initial time (in seconds).

        Returns:
            float: The clock error.
        """
        return (
            self.bias
            + self.drift * np.sin(2 * np.pi * t)
            + np.random.normal(0, self.random_noise)
        )

    def _zero(self) -> None:
        """Set all the errors to zero.

        Returns:
            None
        """
        self.bias = 0
        self.drift = 0
        self.random_noise = 0
        return


class GPSErrorModel:
    """Simulates the behavior of a GPS receiver error naively.

    The error is modelled from a normal distribution with a mean and standard deviation.

    Args:
        bias_r (float): The mean of the receiver error.
        sigma_r (float): The standard deviation of the receiver error.
        bias_sv (float): The mean of the satellite error.
        sigma_sv (float): The standard deviation of the satellite error.
        bias_tropo (float): The mean of the troposphere error.
        sigma_tropo (float): The standard deviation of the troposphere error.
        bias_iono (float): The mean of the ionosphere error.
        sigma_iono (float): The standard deviation of the ionosphere error.
        mutlipath_bias (float): The mean of the multipath error.
        multipath_sigma (float): The standard deviation of the multipath error.
        random_noise (float): The standard deviation of the random walk process.
    """

    def __init__(
        self,
        bias_sv: float = 2,
        sigma_sv: float = 0.1,
        bias_tropo: float = 6,
        sigma_tropo: float = 2,
        bias_iono: float = 8,
        sigma_iono: float = 2,
        mutlipath_bias: float = 3,
        multipath_sigma: float = 0.2,
        cycle_slip_possion_distribution: float = 2,
        random_noise: float = 0.1,
    ) -> None:
        """Initialize the receiver simulator.

        Args:
            bias_r (float): The mean of the receiver error.
            sigma_r (float): The standard deviation of the receiver error.
            bias_sv (float): The mean of the satellite error.
            sigma_sv (float): The standard deviation of the satellite error.
            bias_tropo (float): The mean of the troposphere error.
            sigma_tropo (float): The standard deviation of the troposphere error.
            bias_iono (float): The mean of the ionosphere error.
            sigma_iono (float): The standard deviation of the ionosphere error.
            mutlipath_bias (float): The mean of the multipath error.
            multipath_sigma (float): The standard deviation of the multipath error.
            cycle_slip_possion_distribution (float): The poisson distribution for cycle slips.
            random_noise (float): The standard deviation of the random walk process.

        Returns:
            None
        """
        if sigma_sv < 0.0:
            raise ValueError("Satellite error standard deviation must be non-negative.")
        if sigma_tropo < 0.0:
            raise ValueError(
                "Troposphere error standard deviation must be non-negative."
            )
        if sigma_iono < 0.0:
            raise ValueError(
                "Ionosphere error standard deviation must be non-negative."
            )
        if multipath_sigma < 0.0:
            raise ValueError("Multipath error standard deviation must be non-negative.")
        if random_noise < 0.0:
            raise ValueError("Random noise must be non-negative.")

        # Model troposphere, ionosphere, multipath errors and satellite errors as time varying errors
        self.tropospheric_error = TimeVaryingErrorModel(
            bias=bias_tropo, drift=sigma_tropo, random_noise=random_noise
        )
        self.ionospheric_error = TimeVaryingErrorModel(
            bias=bias_iono, drift=sigma_iono, random_noise=random_noise
        )
        self.multipath_error = TimeVaryingErrorModel(
            bias=mutlipath_bias, drift=multipath_sigma, random_noise=random_noise
        )
        self.satellite_error = TimeVaryingErrorModel(
            bias=bias_sv, drift=sigma_sv, random_noise=random_noise
        )

        # Set the receiver errors
        self.random_noise = random_noise

        # Set the poisson distribution for cycle slips
        self.possion_lambda = cycle_slip_possion_distribution

        return

    def _zero(self) -> None:
        """Set all the errors to zero.

        Returns:
            None
        """
        # Zero the time varying errors
        self.tropospheric_error._zero()
        self.ionospheric_error._zero()
        self.multipath_error._zero()
        self.satellite_error._zero()

        # Zero the receiver errors
        self.random_noise = 0

        # Zero the poisson distribution for cycle slips
        self.possion_lambda = 0

        return

    def get_cycle_slip(self, num_sv: int) -> np.ndarray:
        """Get the cycle slip at a given time.

        Args:
            num_sv (int): Number of satellites.

        Returns:
            np.ndarray: The cycle slip for each satellite.
        """
        return np.array(
            [max(0, np.random.poisson(self.possion_lambda)) for _ in range(num_sv)]
        )

    def get_troposphere_error(self, t: float) -> float:
        """Get the troposphere error at a given time.

        Args:
            t (float): Time elapsed from the initial time (in seconds).

        Returns:
            float: The troposphere error.
        """
        return self.tropospheric_error.get_error(t)

    def get_ionosphere_error(self, t: float) -> float:
        """Get the ionosphere error at a given time.

        Args:
            t (float): Time elapsed from the initial time (in seconds).

        Returns:
            float: The ionosphere error.
        """
        return self.ionospheric_error.get_error(t)

    def get_multipath_error(self, t: float) -> float:
        """Get the multipath error at a given time.

        Args:
            t (float): Time elapsed from the initial time (in seconds).

        Returns:
            float: The multipath error.
        """
        return self.multipath_error.get_error(t)

    def get_satellite_error(self, t: float) -> float:
        """Get the satellite error at a given time.

        Args:
            t (float): Time elapsed from the initial time (in seconds).

        Returns:
            float: The satellite error.
        """
        return self.satellite_error.get_error(t)

    def get_random_noise(self) -> float:
        """Get the random noise at a given time.

        Returns:
            float: The random noise.
        """
        return np.random.normal(0, self.random_noise)
