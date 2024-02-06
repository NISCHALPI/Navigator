"""Unscented Kalman Method for Triangulation.

This module implements the Unscented Kalman Method (UKM) for triangulation, specifically designed for the pseudorange GPS problem.
UKM is a variant of the Kalman Filter that employs the Unscented Transform to estimate the next state vector.
Given the non-linearity in the measurement function of the GPS problem, UKM is preferred over the traditional Kalman Filter. 
The dynamic model assumed here is a constant velocity model.

The state vector is defined as follows:
x = [x, x_dot, y, y_dot, z, z_dot, t, t_dot]

Functions:
- `fx`: State transition function that converts the state vector into the next state vector.
- `hx`: Measurement function for the pseudorange GPS problem, converting the state vector into a pseudorange measurement vector.

Details of the functions:

1. `fx(x: np.ndarray, dt: float) -> np.ndarray`:
    - State transition function.
    - Parameters:
        - x (np.ndarray): The current state vector.
        - dt (float): Time step.
    - Returns:
        np.ndarray: The next state vector.

2. `hx(x: np.ndarray, sv_location: np.ndarray) -> np.ndarray`:
    - Measurement function for the pseudorange GPS problem.
    - Parameters:
        - x (np.ndarray): Current state vector.
        - sv_location (np.ndarray): Location of the satellite.
    - Returns:
        np.ndarray: Pseudorange measurement for the given satellite.


Note:
    Do not optimize the functions using Numba. They become slower.
"""

import numpy as np

from .....utility.transforms.coordinate_transforms import geocentric_to_ellipsoidal
from ....satellite.iephm.sv.tools.elevation_and_azimuthal import elevation_and_azimuthal
from ..algos.troposphere.tropospheric_delay import filtered_troposphere_correction


def fx(x: np.ndarray, dt: float) -> np.ndarray:
    """State transition function.

    Args:
        x (np.ndarray): The state vector.
        dt (float): The time step.

    Returns:
        np.ndarray: The next state vector.
    """
    # The unit velocity transition matrix
    A = A = np.eye(2, dtype=np.float64)
    A[0, 1] = dt

    # The state transition matrix
    X = np.kron(np.eye(4, dtype=np.float64), A)

    # The state transition function
    F = np.eye(9, dtype=np.float64)
    F[:8, :8] = X

    return F @ x


def hx(x: np.ndarray, sv_location: np.ndarray, day_of_year: int) -> np.ndarray:
    """Measurement function for the pseudorange GPS problem.

    Args:
        x (np.ndarray): Current state vector.
        sv_location (np.ndarray): Location of the satellite.
        day_of_year (int): The day of the year. [1-365]

    Returns:
        pseudorange: Pseudorange measurement for the given satellite.
    """
    # Clip the values of the state vector if they are too large
    x = np.clip(x, -1e10, 1e10)

    # Grab the dt from the state vector
    dt = x[6]

    # Grab the tropospheric delay from the state vector
    wet_delay = x[8]

    # Grab the position from the state vector
    position = np.array([x[0], x[2], x[4]], dtype=np.float64)

    # Convert the coordinates from geocentric to ellipsoidal
    lat, _, height = geocentric_to_ellipsoidal(*position)

    # Get the satellite elevation
    E, _ = elevation_and_azimuthal(
        satellite_positions=sv_location,
        observer_position=position,
    )

    # Compute the tropospheric delay
    tropo_delay = [
        filtered_troposphere_correction(
            latitude=lat,
            elevation=E,
            height=height,
            estimated_wet_delay=wet_delay,
            day_of_year=day_of_year,
        )
        for E in E
    ]
    # Compute the pseudorange
    return np.sqrt(np.power((sv_location - position), 2).sum(axis=1)) + dt + tropo_delay
