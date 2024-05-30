"""Module containing the EllipticalTrajectory class."""

import numpy as np

from .trajectory import Trajectory

__all__ = ["EllipticalTrajectory"]


class EllipticalTrajectory(Trajectory):
    """Represents an elliptical trajectory where the car moves along an elliptical path."""

    def __init__(
        self,
        name: str,
        semi_major_axis: float,
        eccentricity: float,
        angular_velocity: float,
        z_perturbation: float = 0.1,
    ) -> None:
        """Initializes the EllipticalTrajectory.

        Args:
            name (str): Name of the trajectory.
            semi_major_axis (float): Semi-major axis of the ellipse.
            eccentricity (float): Eccentricity of the ellipse.
            angular_velocity (float): Angular velocity determining the rate of rotation along the trajectory.
            z_perturbation (float, optional): Perturbation in the z-direction. Defaults to 0.1.
        """
        super().__init__(trajectory_name=name)
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity

        if abs(angular_velocity) < 1e-6:
            raise ValueError("Angular velocity should be non-zero.")

        self.angular_velocity = angular_velocity
        self.z_perturbation = z_perturbation

        # Initial time offset to avoid the car starting at the same position for different trajectories
        self.time_offset = np.random.rand() * 2 * np.pi / angular_velocity

    def get_pos_at_time(self, time: float) -> np.ndarray:
        """Calculates the position of the car at a given time.

        Args:
            time (float): Time at which the position is calculated.

        Returns:
            np.ndarray: Position of the car [x, y, z].
        """
        theta = self.angular_velocity * (time + self.time_offset)
        x = self.semi_major_axis * np.cos(theta)
        y = self.semi_major_axis * np.sin(theta) * np.sqrt(1 - self.eccentricity**2)
        z = self.z_perturbation * np.sin(2 * np.pi * theta)
        return np.array([x, y, z])

    def get_velocity_at_time(self, time: float) -> np.ndarray:
        """Calculates the velocity of the car at a given time.

        Args:
            time (float): Time at which the velocity is calculated.

        Returns:
            np.ndarray: Velocity of the car [vx, vy, vz].
        """
        theta_dot = self.angular_velocity
        theta = self.angular_velocity * (time + self.time_offset)
        x_dot = -self.semi_major_axis * theta_dot * np.sin(theta)
        y_dot = (
            self.semi_major_axis
            * theta_dot
            * np.cos(theta)
            * np.sqrt(1 - self.eccentricity**2)
        )
        z_dot = 0  # Velocity in z-direction is zero
        return np.array([x_dot, y_dot, z_dot])
