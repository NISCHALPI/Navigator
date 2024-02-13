"""These functions are used to calculate the state transition matrix for kalman filter.

Functions:
- `discrete_state_transistion_function`: State transition function that converts the state vector into the next state vector in a discrete time system.

Note:
    Do not optimize the functions using Numba. They become slower.
"""

import numpy as np


def discrete_state_transistion_function(x: np.ndarray, dt: float) -> np.ndarray:
    """State transition function that converts the state vector into the next state vector in a discrete time system.

    State vector:
        x = [x, x_dot, y, y_dot, z, z_dot, t, t_dot]

        where x, y, z are the position coordinates in the ECEF frame, t is the clock bias in meters.

    Args:
        x (np.ndarray): Current state vector.
        dt (float): The sampling time interval in seconds.

    Returns:
        np.ndarray: Next state vector.
    """
    # The unit velocity transition matrix
    A = A = np.eye(2, dtype=np.float64)
    A[0, 1] = dt

    # The state transition matrix
    F = np.kron(np.eye(4, dtype=np.float64), A)

    return F @ x
