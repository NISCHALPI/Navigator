"""This module contains the StationaryTrajectory class."""

import numpy as np

from .trajectory import Trajectory

__all__ = ["StationaryTrajectory"]


class StationaryTrajectory(Trajectory):
    """Represents a stationary trajectory where the car remains at a fixed position."""

    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        """Initializes the StationaryTrajectory.

        Args:
            x (float): x-coordinate of the stationary position.
            y (float): y-coordinate of the stationary position.
            z (float): z-coordinate of the stationary position.
        """
        super().__init__(f"Stationary Trajectory({x}, {y}, {z})")
        self.position = np.array([x, y, z])

    def get_pos_at_time(self, time: float) -> np.ndarray:  # noqa : ARG002
        """Calculates the position of the car at a given time.

        Args:
            time (float): Time at which the position is calculated.

        Returns:
            np.ndarray: Position of the car [x, y, z].
        """
        return self.position

    def get_velocity_at_time(self, time: float) -> np.ndarray:  # noqa : ARG002
        """Calculates the velocity of the car at a given time.

        Args:
            time (float): Time at which the velocity is calculated.

        Returns:
            np.ndarray: Velocity of the car [vx, vy, vz].
        """
        return np.zeros(3)
