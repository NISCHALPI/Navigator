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
import pandas as pd

from ...core.triangulate.itriangulate.algos.combinations.range_combinations import (
    L1_FREQ,
    L1_WAVELENGTH,
    L2_FREQ,
    L2_WAVELENGTH,
    SPEED_OF_LIGHT,
)
from ...epoch.epoch import Epoch
from ..transforms.coordinate_transforms import ellipsoidal_to_geocentric
from .satellite_simulator import GPSConstellation

__all__ = ["RecieverSimulator"]


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


class RecieverSimulator:
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

    RADIUS_OF_EARTH = 6371000  # Radius of the Earth in meters

    constellaion = GPSConstellation()
    error_model = GPSErrorModel()
    clock_model = (
        TimeVaryingErrorModel()
    )  # Already initialized with default values for the clock model

    start_time = pd.Timestamp("2023-01-01 00:00:00")  # Dummy start time

    def __init__(
        self,
        lat: float = 38.8951,  # Latitude of Washington DC
        lon: float = -77.0364,  # Longitude of Washington DC
        alt: float = 72,  # Altitude of Washington DC
        lat_dot: float = 5e-5,
        lon_dot: float = 5e-5,
        alt_dot: float = 0.5,
        lat_noise: float = 0,
        lon_noise: float = 0,
        alt_noise: float = 0,
    ) -> None:
        """Initialize the receiver simulator.

        Args:
            lat (float): Latitude of the receiver in degrees.
            lon (float): Longitude of the receiver in degrees.
            alt (float): Altitude of the receiver in meters.
            lat_dot (float): Latitude velocity of the receiver in degrees per second.
            lon_dot (float): Longitude velocity of the receiver in degrees per second.
            alt_dot (float): Altitude velocity of the receiver in meters per second.
            lat_noise (float): Latitude process noise of the receiver in degrees per second.
            lon_noise (float): Longitude process noise of the receiver in degrees per second.
            alt_noise (float): Altitude process noise of the receiver in meters per second.

        Returns:
            None
        """
        # Check that the noise is non-negative
        if lat_noise < 0.0 or lon_noise < 0.0 or alt_noise < 0.0:
            raise ValueError("Process noise must be non-negative.")

        # Check that lat and lon are within the valid range
        if not (-90 <= lat <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees.")
        if not (-180 <= lon <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees.")

        # Set the initial coordinates
        self.coords = np.array(
            [
                np.radians(lat),
                np.radians(lon),
                alt,
            ]
        )
        self.velocity = np.array([lat_dot, lon_dot, alt_dot])
        self.process_noise = np.array([lat_noise, lon_noise, alt_noise])

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

    def update_position(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Update the position of the receiver based on its velocity.

        Parameters:
        - t (float): Time elapsed from the initial time (in seconds).

        Returns:
        - tuple[np.ndarray, np.ndarray]: Updated coordinates and velocity of the receiver.
        """
        ellipsoidal_location = self.coords + self.velocity * t

        # Add noise to the position
        ellipsoidal_location[0] += np.random.normal(0, self.process_noise[0])
        ellipsoidal_location[1] += np.random.normal(0, self.process_noise[1])
        ellipsoidal_location[2] += np.random.normal(0, self.process_noise[2])

        # Return the updated position in ECEF coordinates
        ecef_position = ellipsoidal_to_geocentric(*ellipsoidal_location)

        # Compute the ellipsoidal velocity in ECEF coordinates using finite difference
        dt = 1e-9  # Time step for finite difference
        ellipsoidal_location_t_plus_dt = self.coords + self.velocity * (t + dt)
        ecef_position_t_plus_dt = ellipsoidal_to_geocentric(
            *ellipsoidal_location_t_plus_dt
        )
        ecef_velocity = (ecef_position_t_plus_dt - ecef_position) / dt

        return ecef_position, ecef_velocity

    def get_epoch(self, t: float, slip: bool = False) -> Epoch:
        """Simulates receiver measurements at a given time.

        Parameters:
        - t (float): Time elapsed from the initial time (in seconds).
        - slip (bool): Whether to regenerate the cycle slips.

        Returns:
        - Epoch: The simulated epoch containing true coordinates and pseudorange measurements from satellites.
        """
        # Update receiver position
        position, velocity = self.update_position(t)

        # Get coordinates of GPS satellites
        satellite_coords = self.constellaion.get_coords(t=t)

        # Add noise to satellite coordinates
        for sat in satellite_coords:
            satellite_coords[sat] += (
                self.error_model.get_satellite_error(t)
                + self.error_model.get_random_noise()
            )

        # Calculate pseudoranges to each satellite
        l1_pseudoranges = {}
        l2_pseudoranges = {}
        l1_phase_measurements = {}
        l2_phase_measurements = {}

        # Get the global errors which is independent of the satellite signal
        clock_error = self.clock_model.get_error(t) * SPEED_OF_LIGHT

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
                + self.error_model.get_random_noise()
            )

            l2_pseudoranges[sat] = (
                range_to_sat
                + clock_error
                + multipath_error[sat]
                + ionosphere_error[sat] * (L2_FREQ / L1_FREQ) ** 2
                + troposphere_error[sat]
                + self.error_model.get_random_noise()
            )
            l1_phase_measurements[sat] = (
                range_to_sat
                + clock_error
                + multipath_error[sat]
                - ionosphere_error[
                    sat
                ]  # The ionosphere error negative as it is a phase measurement
                + troposphere_error[sat]
                + self.error_model.get_random_noise()
                + slips_l1[sat] * L1_WAVELENGTH
            )
            l2_phase_measurements[sat] = (
                range_to_sat
                + clock_error
                + multipath_error[sat]
                - ionosphere_error[sat] * (L2_FREQ / L1_FREQ) ** 2
                + troposphere_error[sat]
                + self.error_model.get_random_noise()
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

        # Create the epoch
        obs_data = pd.DataFrame(
            {
                "C1C": l1_pseudoranges,
                "C2W": l2_pseudoranges,
                "L1C": l1_phase_measurements,
                "L2W": l2_phase_measurements,
            }
        )
        nav_data = pd.DataFrame(sv_coords)

        # Create the epoch
        sim_epoch = Epoch(
            timestamp=self.start_time + pd.Timedelta(seconds=t),
            obs_data=obs_data,
            obs_meta=pd.Series(),
            nav_data=nav_data,
            nav_meta=pd.Series(),
            trim=False,
            purify=False,
            real_coord=truth,
            station="Simulated",
        )

        # Add the dummy profile to the epoch
        sim_epoch.profile = Epoch.DUMMY

        return sim_epoch

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
        self.process_noise = np.zeros(3)

        return

    def __repr__(self) -> str:
        """Return a string representation of the ReceiverSimulator object."""
        return f"ReceiverSimulator(coords={self.coords}, velocity={self.velocity}, process_noise={self.process_noise})"

    def __getitem__(self, key: float) -> Epoch:
        """Get the coordinates of the receiver."""
        return self.get_epoch(key)
