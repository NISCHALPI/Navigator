"""This implements the process noise model for the PPP model."""

import numpy as np

__all__ = ["ppp_process_noise"]


def ppp_process_noise(
    num_sv: int,
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
        curr_state = [x, x_dot , y,  y_dot , z, z_dot , cdt, cdt_dot, t_wet, lambda_N1, ... , lambda_Nn, delta_code1, ... , delta_coden, delta_phase1, ... , delta_phase]

    Args:
        num_sv: The number of satellites to continuously track.
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
    Q = np.eye(9 + num_sv, dtype=np.float64)
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
    # Set other tropospheric delay, integer ambiguity and code and phase bias process noise
    Q[8, 8] = sigma_trop

    # Set the integer ambiguity process noise
    Q[9 : 9 + num_sv, 9 : 9 + num_sv] = np.eye(num_sv, dtype=np.float64) * sigma_bias

    return Q
