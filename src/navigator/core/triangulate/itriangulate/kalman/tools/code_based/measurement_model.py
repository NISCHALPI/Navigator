"""Measurement function for GPS kalman filter.

This module contains the measurement function for the GPS Kalman filter problem. All the measurement functions are to 
be defined in this module.

Measurement Functions:


Note:
    Do not optimize the functions using Numba. They become slower.
"""

import numba as nb
import numpy as np

__all__ = ["measurement_function", "jacobian_measurement_function"]


def measurement_function(state: np.ndarray, sv_location: np.ndarray) -> np.ndarray:
    """Nonlinear measurement function for the GPS Kalman filter problem.

    State vector:
        state = [x, x_dot, y, y_dot, z, z_dot, t, t_dot]

        where x, y, z are the position coordinates in the ECEF frame, t is the clock bias in meters.

    Args:
        state (np.ndarray): Current state vector.
        sv_location (np.ndarray): Location of the satellite in the ECEF frame.(N, 3)

    Returns:
        np.ndarray: Pseudorange measurement for satellite.
    """
    # Grab the range from the state vector
    position = np.array([state[0], state[2], state[4]])
    dt = state[6]

    # Compute the pseudorange
    return np.sqrt(((position - sv_location) ** 2).sum(axis=1)) + dt


def jacobian_measurement_function(
    x: np.ndarray,
    sv_location: np.ndarray,
) -> np.ndarray:
    """Linearized measurement function or observation matrix for the GPS Kalman filter problem used in the EKF.

    State vector:
        x = [x, x_dot, y, y_dot, z, z_dot, t, t_dot]

        where x, y, z are the position coordinates in the ECEF frame, t is the clock bias in meters.


    Args:
        x (np.ndarray): Current state vector.
        sv_location (np.ndarray): Location of the satellite in the ECEF frame.(N, 3)


    Returns:
        np.ndarray: Linearized observation matrix. (N, 8)
    """
    # Grab the range from the state vector
    position = np.array([x[0], x[2], x[4]])

    # # Compute the Jacobian matrix
    # # Initialize the Jacobian matrix
    H = np.zeros((sv_location.shape[0], 8), dtype=np.float64)

    # Compute the pseudorange for each satellite from present state vector
    pseudorange = np.sqrt(((sv_location - position) ** 2).sum(axis=1))

    # Add the range component
    derivative = (position - sv_location) / pseudorange.reshape(-1, 1)

    # Attach the derivative to the Jacobian matrix
    H[:, 0] = derivative[:, 0]
    H[:, 2] = derivative[:, 1]
    H[:, 4] = derivative[:, 2]
    H[:, 6] = 1

    return H
