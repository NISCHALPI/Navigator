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
    F = np.kron(np.eye(4, dtype=np.float64), A)

    return F @ x


def hx(x: np.ndarray, sv_location: np.ndarray) -> np.ndarray:
    """Measurement function for the pseudorange GPS problem.

    Args:
        x (_type_): Current state vector.
        sv_location (_type_): Location of the satellite.

    Returns:
        pseudorange: Pseudorange measurement for the given satellite.
    """
    # Grab the dt from the state vector
    dt = x[6]

    # Grab the position from the state vector
    position = np.array([x[0], x[2], x[4]], dtype=np.float64)

    # Compute the pseudorange
    return np.sqrt(np.power((sv_location - position), 2).sum(axis=1)) + dt
