"""Defines the State Transition Function for a phase-based GPS Kalman Filter.

This module provides functionality for modeling the state transition in a Kalman Filter
for Precise Point Positioning (PPP) using phase-based GPS measurements. It includes the
definition of the state vector and the implementation of the state transition matrix.

State Definition:
    The state vector (curr_state) represents the receiver and is defined as follows:
    x = [latitude, latitude_velocity, longitude, longitude_velocity,
         height, height_velocity, clock_drift, clock_drift_rate,
         wet_tropospheric_delay, B1, ..., Bn]

    Where:
    - latitude: Latitude of the receiver.
    - latitude_velocity: Velocity of the latitude.
    - longitude: Longitude of the receiver.
    - longitude_velocity: Velocity of the longitude.
    - height: Height of the receiver.
    - height_velocity: Velocity of the height.
    - clock_drift: Clock drift.
    - clock_drift_rate: Clock drift rate.
    - wet_tropospheric_delay: Wet tropospheric delay.
    - B: Bias of the phase measurements, including integer ambiguity and hardware delay.

"""

import numba as nb
import numpy as np

__all__ = ["phase_state_transistion_matrix"]


@nb.njit(
    "float64[:, :](float64, int64)",
    fastmath=True,
    parallel=False,
    cache=True,
    error_model="numpy",
    nogil=True,
)
def phase_state_transistion_matrix(
    dt: float,
    num_sv: int,
) -> np.ndarray:
    """State Transition Matrix for the phase-based GPS Kalman Filter.

    Args:
        dt (float): The time step.
        num_sv (int): The number of satellites to continuously track.

    Returns:
        np.ndarray: The state transition matrix for the current state.
    """
    # Constant velocity state transition matrix
    A = np.eye(2, dtype=np.float64)
    A[0, 1] = dt

    F = np.eye(9 + num_sv, dtype=np.float64)
    F_xyz = np.kron(np.eye(4), A)

    # Set the state transition matrix for the ellipsoidal coordinates, clock drift
    F[:8, :8] = F_xyz
    # Other parameters are independent of the state and time hence the state transition matrix is an identity matrix

    return F
