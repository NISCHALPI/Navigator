"""This implements the process noise model for the PPP model."""

import numpy as np

__all__ = ["ppp_process_noise"]


def ppp_process_noise(
    sigma_x: float = 100,
    sigma_y: float = 100,
    sigma_z: float = 100,
    sigma_x_vel: float = 10,
    sigma_y_vel: float = 10,
    sigma_z_vel: float = 10,
    sigma_cdt: float = 100,
    sigma_cdt_dot: float = 10,
    sigma_trop: float = 10,
    sigma_bias: float = 100,
) -> np.ndarray:
    """Process Noise for the PPP model for kalman filter.

    StateDefinition:
        curr_state = [x, x_dot , y,  y_dot , z, z_dot , cdt, cdt_dot, t_wet, lambda_N, delta_code , delta_phase]

    Args:
        sigma_x: The standard deviation of the position in the x direction.
        sigma_y: The standard deviation of the position in the y direction.
        sigma_z: The standard deviation of the position in the z direction.
        sigma_x_vel: The standard deviation of the velocity in the x direction.
        sigma_y_vel: The standard deviation of the velocity in the y direction.
        sigma_z_vel: The standard deviation of the velocity in the z direction.
        sigma_cdt: The standard deviation of the clock drift.
        sigma_cdt_dot: The standard deviation of the clock drift rate.
        sigma_trop: The standard deviation of the tropospheric delay.
        sigma_bias: The standard deviation of the bias.

    Returns:
        The process noise matrix for the PPP model.
    """
    Q = np.zeros((12, 12))
    # Set the position, velocity and the clock drift process noise
    Q[:8, :8] = np.diag(
        [
            sigma_x,
            sigma_y,
            sigma_z,
            sigma_x_vel,
            sigma_y_vel,
            sigma_z_vel,
            sigma_cdt,
            sigma_cdt_dot,
        ]
    )
    # Set other parameters as time invariant
    Q[8:, 8:] = np.diag([sigma_trop, sigma_bias, sigma_bias, sigma_bias])

    return Q
