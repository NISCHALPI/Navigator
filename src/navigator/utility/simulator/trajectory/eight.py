import numpy as np
from .trajectory import Trajectory

__all__ = ["EightTrajctory"]


class EightTrajctory(Trajectory):
    """
    Class for the eight trajectory of the car.

    Trajectory Parametrization:
        r = R * cos(2 * theta)

        in cartesian coordinates:
        x = r * cos(theta)
        y = r * sin(theta)
        z = 0

        the velocity in the x and y directions:
        x_dot = -R * theta_dot (2 * Sin(2 * theta) * Cos(theta) + Cos(2 * theta) * Sin(theta)) # From chain rule on x
        y_dot = R * theta_dot (-2 * Sin(2 * theta) * Sin(theta) + Cos(2 * theta) * Cos(theta)) # From chain rule on y

        Note that the R is constant and the theta_dot is constant given in constructor.
    """

    def __init__(self, radius: float = 10, angular_velocity: float = 1.0):
        """Constructor for the EightTrajectory class.

        Args:
            angular_velocity (float): Angular velocity of the car.

        Returns:
            None
        """
        super().__init__("Eight Trajectory")

        if radius <= 0:
            raise ValueError("Radius should be greater than 0.")

        self.radius = radius
        self.angular_velocity = angular_velocity

    def get_pos_at_time(self, time: float) -> np.ndarray:
        """
        Calculates the position of the car at a given time.

        Parameters:
            time (float): Time at which the position is calculated.

        Returns:
            np.ndarray: Position of the car.
        """
        r, theta = (
            self.radius * np.cos(2 * self.angular_velocity * time),
            self.angular_velocity * time,
        )

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0

        return np.array([x, y, z], dtype=np.float32)

    def get_velocity_at_time(self, time: float) -> np.ndarray:
        """
        Calculates the velocity of the car at a given time.

        Parameters:
            time (float): Time at which the velocity is calculated.

        Returns:
            np.ndarray: Velocity of the car.
        """
        # The car is not moving in the z direction
        Z_DOT = 0
        theta = self.angular_velocity * time

        # Calculate the velocity in the x and y directions
        x_dot = (
            -self.radius
            * self.angular_velocity
            * (
                2 * np.sin(2 * theta) * np.cos(theta)
                + np.cos(2 * theta) * np.sin(theta)
            )
        )
        y_dot = (
            self.radius
            * self.angular_velocity
            * (
                -2 * np.sin(2 * theta) * np.sin(theta)
                + np.cos(2 * theta) * np.cos(theta)
            )
        )

        return np.array([x_dot, y_dot, Z_DOT])
