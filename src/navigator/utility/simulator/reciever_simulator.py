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

from typing import Tuple

import numpy as np
import pandas as pd

from ...core.triangulate.itriangulate.algos.combinations.range_combinations import (
    L1_WAVELENGTH,
    L2_WAVELENGTH,
    SPEED_OF_LIGHT,
    L1_FREQ,
    L2_FREQ,
)
from .satellite_simulator import GPSConstellation

__all__ = ["ReceiverSimulator"]


class NaiveClockModel:
    """Simulates the behavior of a GPS receiver clock naively.

    The clock error is modelled as a simple sinusoidal function of time with a mean and amplitude with random noise
    sampled from a random walk process.


    Args:
        clock_bias (float): The mean of the clock error.
        clock_drift (float): The amplitude of the clock error.
        random_noise (float): The standard deviation of the random walk process.
    """

    def __init__(
        self,
        clock_bias: float = 1e-5,
        clock_drift: float = 3e-6,
        random_noise: float = 1e-6,
    ) -> None:
        """Initialize the clock model.

        Args:
            clock_bias (float): The mean of the clock error.
            clock_drift (float): The amplitude of the clock error.
            random_noise (float): The standard deviation of the random walk process.

        Returns:
            None
        """
        if clock_bias < 0.0:
            raise ValueError("Clock bias must be non-negative.")
        if clock_drift < 0.0:
            raise ValueError("Clock drift must be non-negative.")

        self.clock_bias = clock_bias
        self.clock_drift = clock_drift

        if random_noise < 0.0:
            raise ValueError("Random noise must be non-negative.")

        self.random_noise = random_noise

    def get_clock_error(self, t: float) -> float:
        """Get the clock error at a given time.

        Args:
            t (float): Time elapsed from the initial time (in seconds).

        Returns:
            float: The clock error.
        """
        return (
            self.clock_bias
            + self.clock_drift * np.sin(2 * np.pi * t)
            + np.random.normal(0, self.random_noise)
        )

    def _zero(self) -> None:
        """Set all the errors to zero.

        Returns:
            None
        """
        self.clock_bias = 0
        self.clock_drift = 0
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
        bias_sv: float = 1,
        sigma_sv: float = 0.5,
        bias_tropo: float = 6,
        sigma_tropo: float = 0.5,
        bias_iono: float = 9,
        sigma_iono: float = 0.5,
        mutlipath_bias: float = 3,
        multipath_sigma: float = 0.1,
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
            cycle_slip_lambda (float): The coefficient of the poisson distribution for cycle slips.
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

        self.bias_sv = bias_sv
        self.sigma_sv = sigma_sv

        self.bias_tropo = bias_tropo
        self.sigma_tropo = sigma_tropo
        self.bias_iono = bias_iono
        self.sigma_iono = sigma_iono
        self.mutlipath_bias = mutlipath_bias

        self.multipath_sigma = multipath_sigma
        self.random_noise = random_noise

        self.possion_lambda = cycle_slip_possion_distribution

        return

    def _zero(self) -> None:
        """Set all the errors to zero.

        Returns:
            None
        """
        self.bias_sv = 0
        self.sigma_sv = 0

        self.bias_tropo = 0
        self.sigma_tropo = 0
        self.bias_iono = 0
        self.sigma_iono = 0
        self.mutlipath_bias = 0

        self.multipath_sigma = 0
        self.random_noise = 0

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
        return np.random.normal(self.bias_tropo, self.sigma_tropo)

    def get_ionosphere_error(self, t: float) -> float:
        """Get the ionosphere error at a given time.

        Args:
            t (float): Time elapsed from the initial time (in seconds).

        Returns:
            float: The ionosphere error.
        """
        return np.random.normal(self.bias_iono, self.sigma_iono)

    def get_multipath_error(self, t: float) -> float:
        """Get the multipath error at a given time.

        Args:
            t (float): Time elapsed from the initial time (in seconds).

        Returns:
            float: The multipath error.
        """
        return np.random.normal(self.mutlipath_bias, self.multipath_sigma)

    def get_satellite_error(self, t: float) -> float:
        """Get the satellite error at a given time.

        Args:
            t (float): Time elapsed from the initial time (in seconds).

        Returns:
            float: The satellite error.
        """
        return np.random.normal(self.bias_sv, self.sigma_sv)

    def get_random_noise(self, t: float) -> float:
        """Get the random noise at a given time.

        Args:
            t (float): Time elapsed from the initial time (in seconds).

        Returns:
            float: The random noise.
        """
        return np.random.normal(0, self.random_noise)


class ReceiverSimulator:
    """Simulates the behavior of a GPS receiver, including position updates and pseudorange measurements.

    Attributes:
        constellaion (GPSConstellation): GPS constellation instance used by the ReceiverSimulator.

    Methods:
        __init__(self, initial_coords, velocity, mean_psuedorange_error, std_psuedorange_error, mean_satellite_error, std_satellite_error):
            Initializes the ReceiverSimulator object with specified parameters.

        update_position(self, t):
            Updates the position of the receiver based on its velocity.

            Parameters:
            - t (float): Time elapsed from the initial time (in seconds).

    Returns:
            - np.ndarray: Updated coordinates of the receiver.

        get_measurements(self, t):
            Simulates receiver measurements at a given time, including true coordinates and pseudorange measurements.

            Parameters:
            - t (float): Time elapsed from the initial time (in seconds).

    Returns:
            - Tuple[pd.Series, pd.DataFrame]: True coordinates of the receiver, and pseudorange measurements from satellites.

    Note:
        The parameters for initialization (initial_coords, velocity, mean_psuedorange_error, etc.) should be adjusted based on the specific scenario you want to simulate.
    """

    constellaion = GPSConstellation()
    error_model = GPSErrorModel()
    clock_model = NaiveClockModel()

    def __init__(
        self,
        coords: np.ndarray = np.array(
            [1115077.69025948, -4843958.49112974, 3983260.99261736]
        ),
        velocity: np.ndarray = np.array([5, 5, 5]),
        process_noise: float = 1,
    ) -> None:
        """Initialize the receiver simulator.

        Args:
            coords (np.ndarray): Initial coordinates of the receiver.
            velocity (np.ndarray): Velocity of the receiver.
            process_noise (float): Standard deviation of the process noise.

        Returns:
            None
        """
        if len(coords) != 3:
            raise ValueError("Coordinates must be a 3D vector.")
        if len(velocity) != 3:
            raise ValueError("Velocity must be a 3D vector.")

        self.coords = coords
        self.velocity = velocity
        self.process_noise = process_noise

        # Initalize the cycle slips
        self.cycle_slip()

    def cycle_slip(self) -> None:
        """Simulate a cycle slip in the receivr, Aer.

        Returns:
            None
        """
        self.slips_l1 = self.error_model.get_cycle_slip(
            len(self.constellaion.satellites)
        )
        self.slips_l2 = self.error_model.get_cycle_slip(
            len(self.constellaion.satellites)
        )

    def update_position(self, t: float) -> np.ndarray:
        """Update the position of the receiver based on its velocity.

        Parameters:
        - t (float): Time elapsed from the initial time (in seconds).

        Returns:
        - np.ndarray: Updated coordinates of the receiver.
        """
        return (
            self.coords
            + self.velocity * t
            + np.random.normal(0, self.process_noise, 3),
            self.velocity,
        )

    def get_measurements(
        self, t: float, slip: bool = False
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Simulates receiver measurements at a given time.

        Parameters:
        - t (float): Time elapsed from the initial time (in seconds).
        - slip (bool): Whether to regenerate the cycle slips.

        Returns:
        - Tuple[pd.Series, pd.DataFrame]: True coordinates of the receiver and velocity of, and pseudorange measurements from satellites.
        """
        # Update receiver position
        position, velocity = self.update_position(t)

        # Get coordinates of GPS satellites
        satellite_coords = self.constellaion.get_coords(t=t)

        # Add noise to satellite coordinates
        for sat in satellite_coords:
            satellite_coords[sat] += self.error_model.get_satellite_error(
                t
            ) + self.error_model.get_random_noise(t)

        # Calculate pseudoranges to each satellite
        l1_pseudoranges = {}
        l2_pseudoranges = {}
        l1_phase_measurements = {}
        l2_phase_measurements = {}

        # Get the global errors which is independent of the satellite signal
        clock_error = self.clock_model.get_clock_error(t) * SPEED_OF_LIGHT

        for sat, coords in satellite_coords.items():
            # Get the true range to the satellite
            range_to_sat = np.linalg.norm(coords - position)

            # Get satellite specific errors to each satellite signal i.e multipath, ionosphere, troposphere
            multipath_error = {
                sv: self.error_model.get_multipath_error(t)
                for sv in self.constellaion.satellites
            }
            ionosphere_error = {
                sv: self.error_model.get_ionosphere_error(t)
                for sv in self.constellaion.satellites
            }
            troposphere_error = {
                sv: self.error_model.get_troposphere_error(t)
                for sv in self.constellaion.satellites
            }

            # Get the cycle slip for the satellite
            if slip:
                self.cycle_slip()
            slips_l1 = {
                sv: self.slips_l1[i]
                for i, sv in enumerate(self.constellaion.satellites)
            }
            slips_l2 = {
                sv: self.slips_l2[i]
                for i, sv in enumerate(self.constellaion.satellites)
            }

            # Compute the l1 pseudorange
            l1_pseudoranges[sat] = (
                range_to_sat
                + clock_error
                + multipath_error[sat]
                + ionosphere_error[sat]
                + troposphere_error[sat]
                + self.error_model.get_random_noise(t)
            )

            l2_pseudoranges[sat] = (
                range_to_sat
                + clock_error
                + multipath_error[sat]
                + ionosphere_error[sat] * (L2_FREQ / L1_FREQ) ** 2
                + troposphere_error[sat]
                + self.error_model.get_random_noise(t)
            )
            l1_phase_measurements[sat] = (
                range_to_sat
                + clock_error
                + multipath_error[sat]
                - ionosphere_error[
                    sat
                ]  # The ionosphere error negative as it is a phase measurement
                + troposphere_error[sat]
                + self.error_model.get_random_noise(t)
                + slips_l1[sat] * L1_WAVELENGTH
            )
            l2_phase_measurements[sat] = (
                range_to_sat
                + clock_error
                + multipath_error[sat]
                - ionosphere_error[sat] * (L2_FREQ / L1_FREQ) ** 2
                + troposphere_error[sat]
                + self.error_model.get_random_noise(t)
                + slips_l2[sat] * L2_WAVELENGTH
            )

        # Return measurements
        truth = {
            "x": position[0],
            "y": position[1],
            "z": position[2],
            "x_dot": velocity[0],
            "y_dot": velocity[1],
            "z_dot": velocity[2],
            "cdt": clock_error,
            "tropo": troposphere_error,
            "iono": ionosphere_error,
            "multipath": multipath_error,
            "slips_l1": slips_l1,
            "slips_l2": slips_l2,
        }

        # Get the sv measurements
        sv_coords = pd.DataFrame(satellite_coords).T.rename(
            columns={0: "x", 1: "y", 2: "z"}
        )

        # Get the measurements
        l1_pseudoranges = pd.Series(l1_pseudoranges)
        l2_pseudoranges = pd.Series(l2_pseudoranges)
        l1_phase_measurements = pd.Series(l1_phase_measurements)
        l2_phase_measurements = pd.Series(l2_phase_measurements)

        # Add measurements to a dataframe
        sv_coords["C1C"] = l1_pseudoranges
        sv_coords["C2W"] = l2_pseudoranges
        sv_coords["L1C"] = l1_phase_measurements
        sv_coords["L2W"] = l2_phase_measurements

        return pd.Series(truth), sv_coords

    def _zero(self) -> None:
        """Set all the errors to zero.

        Returns:
            None
        """
        # Set the internal errors to zero
        self.error_model._zero()
        self.clock_model._zero()

        # Set the cycle slips to zero
        self.slips_l1 = np.zeros(len(self.constellaion.satellites))
        self.slips_l2 = np.zeros(len(self.constellaion.satellites))

        # Set the process noise to zero
        self.process_noise = 0
        return
