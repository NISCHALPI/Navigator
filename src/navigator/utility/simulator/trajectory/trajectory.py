"""This module contains the abstract class for the trajectory of the car."""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

__all__ = [
    "Trajectory",
]


class Trajectory(ABC):
    """Abstract class for the trajectory of the car."""

    def __init__(self, trajectory_name: str, scale: float = 1) -> None:
        """Constructor for the Trajectory class.

        Args:
            trajectory_name (str): Name of the trajectory.
            scale (float, optional): Scale factor for the trajectory. Defaults to 1.

        Returns:
            None
        """
        self.trajectory_name = trajectory_name
        self.scale = scale

    @abstractmethod
    def get_pos_at_time(self, time: float) -> np.ndarray:
        """Calculates the position of the car at a given time.

        Parameters:
            time (float): Time at which the position is calculated.

        Returns:
            np.ndarray: Position of the car.
        """
        pass

    @abstractmethod
    def get_velocity_at_time(self, time: float) -> np.ndarray:
        """Calculates the velocity of the car at a given time.

        Parameters:
            time (float): Time at which the velocity is calculated.

        Returns:
            np.ndarray: Velocity of the car.
        """
        pass

    def __call__(self, time: float) -> np.ndarray:
        """Calls the get_pos_at_time method to calculate the position of the car at a given time.

        Parameters:
            time (float): Time at which the position is calculated.

        Returns:
            np.ndarray: Position and velocity of the car.
        """
        pos = self.get_pos_at_time(time)
        vel = self.get_velocity_at_time(time)

        return np.concatenate([pos, vel], axis=0) * self.scale

    def __repr__(self) -> str:
        """Returns the string representation of the Trajectory object.

        Returns:
            str: String representation of the Trajectory object.
        """
        return f"Trajectory({self.trajectory_name})"

    def __mul__(self, scale: float) -> "Trajectory":
        """Scales the trajectory by a factor.

        Args:
            scale (float): The scale factor.

        Returns:
            Trajectory: The scaled trajectory.
        """
        clone = deepcopy(self)
        clone.scale *= scale

        return clone
