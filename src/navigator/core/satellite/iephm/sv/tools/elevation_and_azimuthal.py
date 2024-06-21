"""Module for calculating satellite elevation and azimuthal angles.

This module provides a function, elevation_and_azimuthal, to determine the elevation
and azimuthal angles of a satellite relative to an observer on Earth.

Dependencies:
  - numpy: For numerical operations and array manipulations.
  - geocentric_to_enu: Function for converting geocentric coordinates to local ENU coordinates.

Usage:
  Call the elevation_and_azimuthal function with satellite and observer ECEF coordinates
  to obtain the elevation and azimuthal angles in degrees.
"""

import numpy as np  # type: ignore

from ......utils.transforms.coordinate_transforms import (
    geocentric_to_enu,
)

__all__ = ["elevation_and_azimuthal"]


def elevation_and_azimuthal(
    satellite_positions: np.ndarray, observer_position: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the elevation and azimuthal angles of multiple satellites.

    Args:
        satellite_positions (np.ndarray): The positions of the satellites in ECEF coordinates. Shape=(N, 3)
        observer_position (np.ndarray): The position of the observer in ECEF coordinates. Shape=(3,)

    Returns:
        tuple: A tuple containing the elevation and azimuthal angles of the satellites. (E, A)
    """
    # Calculate the local ENU coordinates of the observer
    enu = geocentric_to_enu(
        observer_position[0], observer_position[1], observer_position[2]
    )

    # Calculate the LOS vectors from the observer to the satellites
    los = np.float32(satellite_positions - observer_position)
    los = los / np.linalg.norm(los, axis=1, keepdims=True)

    # Calculate the elevation angle
    E = np.arcsin(np.dot(los, enu[2]))

    # Calculate the azimuthal angle
    A = np.arctan2(np.dot(los, enu[0]), np.dot(los, enu[1]))

    # Convert the angles to degrees
    E = np.degrees(E)
    A = np.degrees(A)

    # Wrap the azimuthal angles to the range [0, 360)
    A = (A + 360) % 360
    # Wrap the elevation angles to the range [-90, 90]
    E = (E + 90) % 180 - 90

    return E, A
