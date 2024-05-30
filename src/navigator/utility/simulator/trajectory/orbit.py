"""This module contains the implementation of the SatelliteLikeTrajectory class."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .ellipsoidal import EllipticalTrajectory
from .trajectory import Trajectory


class SatelliteLikeTrajectory(Trajectory):
    """Represents a satellite-like trajectory which is parametrized by the kepelian elements.

    Parameters:
        name (str): Name of the trajectory.
        semi_major_axis (float): Semi-major axis of the ellipse.
        eccentricity (float): Eccentricity of the ellipse.
        inclination (float): Inclination of the orbit.
        right_ascension (float): Right ascension of the ascending node.
        argument_perigee (float): Argument of perigee.
        true_anomaly (float): True anomaly.
        angular_velocity (float): Angular velocity determining the rate of rotation along the trajectory.
    """

    def __init__(
        self,
        name: str,
        semi_major_axis: float,
        eccentricity: float,
        inclination: float,
        right_ascension: float,
        argument_perigee: float,
        true_anomaly: float,
        angular_velocity: float,
    ) -> None:
        """Initializes the SatelliteLikeTrajectory object.

        The arguments are the kepelian elements that define the orbit of the satellite in space.
        The units are in meters and radians.

        Args:
            name (str): Name of the satellite.
            semi_major_axis (float): Semi-major axis of the ellipse.
            eccentricity (float): Eccentricity of the ellipse.
            inclination (float): Inclination of the orbit.
            right_ascension (float): Right ascension of the ascending node.
            argument_perigee (float): Argument of perigee.
            true_anomaly (float): True anomaly.
            angular_velocity (float): Angular velocity determining the rate of rotation along the trajectory.
        """
        # Initialize the Trajectory object
        super().__init__(name)

        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.right_ascension = right_ascension
        self.argument_perigee = argument_perigee
        self.true_anomaly = true_anomaly
        self.angular_velocity = angular_velocity

        # Elliptical orbit parameters
        self.elliptical_trajectory = EllipticalTrajectory(
            name=name,
            semi_major_axis=semi_major_axis,
            eccentricity=eccentricity,
            angular_velocity=angular_velocity,
        )

        # Rotation matrix to transform the position flat xy-plane to the orbital plane
        self.rotation_matrix = R.from_euler(
            "zyz",
            [right_ascension, inclination, argument_perigee],
            degrees=False,
        ).as_matrix()

    def get_pos_at_time(self, time: float) -> np.ndarray:
        """Calculates the position of the satellite at a given time.

        Args:
            time (float): Time at which the position is calculated.

        Returns:
            np.ndarray: Position of the satellite [x, y, z].
        """
        # Calculate the position in the elliptical orbit
        pos = self.elliptical_trajectory.get_pos_at_time(
            self.true_anomaly / self.angular_velocity + time
        )
        # Rotate the position to the orbital plane
        return self.rotation_matrix @ pos

    def get_velocity_at_time(self, time: float) -> np.ndarray:
        """Calculates the velocity of the satellite at a given time.

        Args:
            time (float): Time at which the velocity is calculated.

        Returns:
            np.ndarray: Velocity of the satellite [vx, vy, vz].
        """
        # Calculate the velocity in the elliptical orbit
        vel = self.elliptical_trajectory.get_velocity_at_time(
            self.true_anomaly / self.angular_velocity + time
        )
        # Rotate the velocity to the orbital plane
        return self.rotation_matrix @ vel
