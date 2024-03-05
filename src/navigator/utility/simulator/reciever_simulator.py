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
from .satellite_simulator import GPSConstellation

__all__ = ["ReceiverSimulator"]


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

    def __init__(
        self,
        initial_coords: Tuple[float, float, float] = (
            1115077.69025948,
            -4843958.49112974,
            3983260.99261736,
        ),  # Washington DC Coordinates
        velocity: Tuple[float, float, float] = (5, 5, 5),
        mean_psuedorange_error: float = 10,
        std_psuedorange_error: float = 5,
        mean_satellite_error: float = 1,
        std_satellite_error: float = 1,
    ) -> None:
        """Initialize the receiver simulator.

        Parameters:
        - initial_coords (tuple): Initial coordinates of the receiver (x, y, z).
        - velocity (tuple): Initial velocity of the receiver (vx, vy, vz).
        - mean_psuedorange_error (float): Mean of the pseudorange error distribution.
        - std_psuedorange_error (float): Standard deviation of the pseudorange error distribution.
        - mean_satellite_error (float): Mean of the satellite error distribution.
        - std_satellite_error (float): Standard deviation of the satellite error distribution.
        """
        self.coords = np.array(initial_coords)
        self.velocity = np.array(velocity)

        self.mean_psuedorange_error = mean_psuedorange_error
        self.std_psuedorange_error = std_psuedorange_error
        self.mean_satellite_error = mean_satellite_error
        self.std_satellite_error = std_satellite_error

    def update_position(self, t: float) -> np.ndarray:
        """Update the position of the receiver based on its velocity.

        Parameters:
        - t (float): Time elapsed from the initial time (in seconds).

        Returns:
        - np.ndarray: Updated coordinates of the receiver.
        """
        return self.coords + self.velocity * t

    def get_measurements(self, t: float) -> Tuple[pd.Series, pd.DataFrame]:
        """Simulates receiver measurements at a given time.

        Parameters:
        - t (float): Time elapsed from the initial time (in seconds).

        Returns:
        - Tuple[pd.Series, pd.DataFrame]: True coordinates of the receiver, and pseudorange measurements from satellites.
        """
        # Update receiver position
        position = self.update_position(t)

        # Get coordinates of GPS satellites
        satellite_coords = self.constellaion.get_coords(t=t)

        # Add noise to satellite coordinates
        for sat in satellite_coords:
            satellite_coords[sat] += np.random.normal(
                self.mean_satellite_error, self.std_satellite_error, 3
            )

        # Calculate pseudoranges to each satellite
        pseudoranges = {}
        for sat, coords in satellite_coords.items():
            pseudoranges[sat] = np.linalg.norm(position - coords) + np.random.normal(
                self.mean_psuedorange_error, self.std_psuedorange_error
            )

        # Return measurements
        true_coords = pd.Series(position, index=["x", "y", "z"])
        sv_coords = pd.DataFrame(satellite_coords).T.rename(
            columns={0: "x", 1: "y", 2: "z"}
        )
        pseudoranges = pd.Series(pseudoranges, name="pseudorange")

        sv_coords["pseudorange"] = pseudoranges

        return true_coords, sv_coords
