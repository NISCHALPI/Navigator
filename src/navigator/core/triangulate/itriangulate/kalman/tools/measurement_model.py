"""Measurement function for GPS kalman filter.

This module contains the measurement function for the GPS Kalman filter problem. All the measurement functions are to 
be defined in this module.

Measurement Functions:


Note:
    Do not optimize the functions using Numba. They become slower.
"""
import numpy as np

__all__ = ["measurement_function", "jacobian_measurement_function"]


def measurement_function(x: np.ndarray, sv_location: np.ndarray) -> np.ndarray:
    """Nonlinear measurement function for the GPS Kalman filter problem.

    State vector:
        x = [x, x_dot, y, y_dot, z, z_dot, t, t_dot]

        where x, y, z are the position coordinates in the ECEF frame, t is the clock bias in meters.

    Args:
        x (np.ndarray): Current state vector.
        sv_location (np.ndarray): Location of the satellite in the ECEF frame.(N, 3)

    Returns:
        np.ndarray: Pseudorange measurement for satellite.
    """
    # Grab the dt from the state vector
    position = x[[0, 2, 4]]
    dt = x[6]

    # Compute the pseudorange
    return np.sqrt(np.power((sv_location - position), 2).sum(axis=1)) + dt


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
    position = x[:3]
    x[6]

    # Compute the Jacobian matrix
    # Initialize the Jacobian matrix
    H = np.zeros((len(sv_location), 8), dtype=np.float64)
    # Compute the pseudorange for each satellite from present state vector
    pseudorange = np.sqrt(np.power((sv_location - position), 2).sum(axis=1))

    # Add the range component
    H[:, [0, 2, 4]] = (position - sv_location) / pseudorange[:, None]
    # Add the clock bias component
    H[:, 6] = 1

    return H
