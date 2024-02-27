"""Defines the State Transition Function for the PPP modes.

This model is only defined for the GPS PPP model. This is not multi-GNSS model supported.


StateDefinition:
    curr_state = [x, x_dot , y,  y_dot , z, z_dot , cdt, cdt_dot, t_wet, lambda_N1, ... , lambda_Nn, delta_code1, ... , delta_coden, delta_phase1, ... , delta_phase]
    where:
        x, y, z: are the ECEF coordinates of the receiver.
        x_dot, y_dot, z_dot: are the velocities of the receiver.
        t_wet: is the wet tropospheric delay.
        lambda_N: is the integer ambiguity for the phase measurements.
"""

import numba as nb
import numpy as np

__all__ = ["ppp_state_transistion_matrix"]


@nb.njit(
    "float64[:, :](float64, int64)",
    fastmath=True,
    parallel=False,
    cache=True,
    error_model="numpy",
    nogil=True,
)
def ppp_state_transistion_matrix(
    dt: float,
    num_sv: int,
) -> np.ndarray:
    """State Transition Function for the PPP model for kalman filter.

    Args:
        state: The current state of the system.
        dt: The time step.
        num_sv: The number of satellites to continuously track.

    StateDefinition:
        curr_state = [x, x_dot , y,  y_dot , z, z_dot , cdt, cdt_dot, t_wet, lambda_N1, ... , lambda_Nn, delta_code1, ... , delta_coden, delta_phase1, ... , delta_phase]
        where:
            x, y, z: are the ECEF coordinates of the receiver.
            x_dot, y_dot, z_dot: are the velocities of the receiver.
            t_wet: is the wet tropospheric delay.
            lambda_N: is the integer ambiguity for the phase measurements.

    Returns:
        The state transition matrix for the current state.
    """
    # Constant velocity state transition matrix
    A = np.eye(2, dtype=np.float64)
    A[0, 1] = dt

    F = np.eye(9 + 1 * num_sv, dtype=np.float64)
    F_xyz = np.kron(np.eye(4), A)

    # Set the position, velocity and the clock drift state transition matrix
    F[:8, :8] = F_xyz

    return F
