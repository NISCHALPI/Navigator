"""Contains Rotation Matrices for Earth Rotation Effects in Satellite Position Calculations.

This module provides rotation matrices for correcting satellite positions due to Earth's rotation. Satellite positions are calculated in Earth-Centered, Earth-Fixed (ECEF) coordinates at a time in the past due to light time equations. 
The rotation matrices are used to adjust the satellite position to reciver current coordinate system.

Constants:
    OMEGA_EARTH (float): Earth's rotation rate in radians per second.
    c (float): Speed of light in meters per second.

Functions:
    earth_rotation_correction(sv_position: np.ndarray, delta_t: float) -> np.ndarray:
        Calculate the corrected satellite position accounting for Earth's rotation.

Args:
    sv_position (np.ndarray): A (3,1) numpy array representing the satellite's ECEF position.
    delta_t (float): Time interval in seconds between the past position and the current time.

Returns:
    np.ndarray: A (3,1) numpy array representing the corrected satellite position after Earth's rotation adjustment.

Summary:
    This module contains functionality to calculate the corrected satellite position by applying rotation matrices that account for Earth's rotation. The `earth_rotation_correction` function takes the satellite's ECEF position and a time interval (delta_t) as input and returns the corrected position to account for the time difference between calculation and the current time.
"""
import warnings

import numpy as np
from numba import float64, njit, prange
from numba.core.errors import NumbaPerformanceWarning

# Filter out the numba warnings for this module
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


# Constants
OMEGA_EARTH = 7.2921151467e-5  # Earth's rotation rate in rad/s
c = 299792458.0  # Speed of light in m/s


@njit(float64[:, :](float64), cache=True)
def _rotation_matrix(theta: float) -> np.ndarray:
    """Return the rotation matrix for a rotation of theta radians.

    Args:
        theta (float): The angle of rotation in radians.

    Returns:
        np.ndarray: A (3,3) numpy array representing the rotation matrix.
    """
    # Calculate the rotation matrix
    rotation_matrix = np.eye(3, dtype=np.float64)

    # Add The rotation values
    rotation_matrix[0, 0] = np.cos(theta)
    rotation_matrix[0, 1] = np.sin(theta)
    rotation_matrix[1, 0] = -np.sin(theta)
    rotation_matrix[1, 1] = np.cos(theta)

    return rotation_matrix


@njit(float64[:, :](float64[:, :], float64[:]), parallel=True, cache=True)
def earth_rotation_correction(
    sv_position: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """Calculate the corrected satellite position accounting for Earth's rotation.

    Args:
       sv_position (np.ndarray): Must be (num_sv, 3) numpy array representing the satellite's ECEF position(s
       dt (float): Must be of shape  (num_sv) numpy array representing the time interval in seconds between the past position and the current time. (num_sv, 1

    Returns:
       np.ndarray: A (3,1) numpy array representing the corrected satellite position after Earth's rotation adjustment.
    """
    # Check that the input is a numpy array
    assert sv_position.shape[-1] == 3, "Input must be a (num_sv, 3) numpy array"
    assert dt.shape == (sv_position.shape[0],), "Input must be a (num_sv) numpy array"

    # Apply the rotation matrix to each satellite position
    for sv in prange(sv_position.shape[0]):
        earth_rotation = _rotation_matrix(OMEGA_EARTH * dt[sv])
        sv_position[sv] = earth_rotation @ sv_position[sv]

    return sv_position
