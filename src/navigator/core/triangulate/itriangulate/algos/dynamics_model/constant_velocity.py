"""Implements constant velocity dynamics model for kalman filter for triangulation.

In the contex of triangulation, the dynamics model is used to predict the future state 
estimates of the target. We define the state vector as:


x = [x, x_dot, y, y_dot, z, z_dot, cdt, cdt_dot]


where x, y, z are the position of the target in the ECEF frame and x_dot, y_dot, z_dot are the
velocities in the ECEF frame. cdt is the clock drift of the target and cdt_dot is the rate of change.

"""

import numpy as np

__all__ = ["G", "hx", "HJacobian", "Q"]


# Constant Velocity State Transition Matrix
def G(dt: float) -> np.ndarray:
    """Returns the state transition matrix for the constant velocity model.

    Args:
        dt (float): Time step.

    Returns:
        np.ndarray: State transition matrix.
    """
    return np.array(
        [
            [1, dt, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, dt, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def hx(x: np.ndarray, sv_pos: np.ndarray) -> np.ndarray:
    """Returns the measurement matrix for the constant velocity model.

    Args:
        x (np.ndarray): State vector.
        sv_pos (np.ndarray): Satellite position.

    Returns:
        np.ndarray: Measurement matrix.
    """
    pos = x[[0, 2, 4]]
    return np.linalg.norm(pos - sv_pos, axis=1) + x[6]


def HJacobian(x: np.ndarray, sv_pos: np.ndarray) -> np.ndarray:
    """Returns the jacobian of the measurement matrix for the constant velocity model.

    Args:
        x (np.ndarray): State vector.
        sv_pos (np.ndarray): Satellite position.

    Returns:
        np.ndarray: Jacobian of the measurement matrix.
    """
    pos = x[[0, 2, 4]]
    diff = pos - sv_pos
    norm = np.linalg.norm(diff, axis=1)

    # Initialize the jacobian matrix
    HJ = np.zeros((sv_pos.shape[0], 8))

    # Add the derivative of the measurement matrix wrt position
    HJ[:, [0, 2, 4]] = diff / norm[:, None]

    # Add the derivative of the measurement matrix wrt clock bias
    HJ[:, 6] = 1

    return HJ


def Q(dt: float, autocorrelation: np.ndarray) -> np.ndarray:
    """Returns the process noise matrix for the constant velocity model.

    Args:
        dt (float): Time step.
        autocorrelation (np.ndarray): Autocorrelation matrix of the process noise (AWGN) (8x8 matrix)

    Returns:
        np.ndarray: Process noise matrix.
    """
    A = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]], dtype=np.float64)

    F = np.kron(np.eye(4), A)

    return F @ autocorrelation @ F.T
