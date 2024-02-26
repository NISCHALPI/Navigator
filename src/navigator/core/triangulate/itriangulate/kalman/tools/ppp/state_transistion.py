"""Defines the State Transition Function for the PPP modes.

This model is only defined for the GPS PPP model. This is not multi-GNSS model supported.


StateDefinition:
    curr_state = [x, x_dot , y,  y_dot , z, z_dot , cdt, cdt_dot, t_wet, lambda_N, delta_code , delta_phase]
    where:
        x, y, z: are the ECEF coordinates of the receiver.
        x_dot, y_dot, z_dot: are the velocities of the receiver.
        t_wet: is the wet tropospheric delay.
        lambda_N: is the integer ambiguity for the phase measurements.
"""

import numpy as np

from ..spp.state_transistion import constant_velocity_state_transistion

__all__ = ["ppp_state_transistion_matrix"]


def ppp_state_transistion_matrix(
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """State Transition Function for the PPP model for kalman filter.

    Args:
        state: The current state of the system.
        dt: The time step.

    StateDefinition:
        curr_state = [x, x_dot , y,  y_dot , z, z_dot , cdt, cdt_dot, t_wet, lambda_N, delta_code , delta_phase]
        where:
            x, y, z: are the ECEF coordinates of the receiver.
            x_dot, y_dot, z_dot: are the velocities of the receiver.
            t_wet: is the wet tropospheric delay.
            lambda_N: is the integer ambiguity for the phase measurements.

    Returns:
        The state transition matrix for the current state.
    """
    # Create a block diagonal matrix of size 12x12 based on the state
    F = np.zeros((12, 12))
    F_xyz = constant_velocity_state_transistion(x=np.eye(8), dt=dt)

    # Set the position, velocity and the clock drift state transition matrix
    F[:8, :8] = F_xyz
    # Set other parameters as time invariant
    F[8:, 8:] = np.eye(4)

    return F @ state
