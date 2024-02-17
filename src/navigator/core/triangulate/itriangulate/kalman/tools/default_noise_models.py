"""This module contains the default noise models for the Kalman Filter.

The noise models are used to define the process and measurement noise covariance matrices for the Kalman Filter.



"""

import numpy as np

from .noise_profile import (
    clock_process_noise_profile,
    diagonal_measurement_noise_profile,
    dynamic_position_process_noise_profile,
)


def measurement_noise_profile(sigma_r: float, n_measurement: int) -> np.ndarray:
    """The measurement noise profile.

    Args:
        sigma_r (float): The measurement noise.
        n_measurement (int): The number of measurements.

    Returns:
        np.ndarray: The measurement noise covariance matrix.
    """
    return diagonal_measurement_noise_profile(
        sigma_r=sigma_r, n_measurement=n_measurement
    )


def octa_state_process_noise_profile(
    S_x: float,
    S_y: float,
    S_z: float,
    dt: float,
    h_0: float = 2e-19,
    h_2: float = 2e-20,
) -> np.ndarray:
    """The process noise profile for the Octa-State vector.

    State vector:
        [x, x_dot, y, y_dot, z, z_dot, clock_bias, clock_drift]

    Args:
        S_x (float): The white noise spectral density for the random walk position error in the x-direction.
        S_y (float): The white noise spectral density for the random walk position error in the y-direction.
        S_z (float): The white noise spectral density for the random walk position error in the z-direction.
        dt (float): The sampling time interval in seconds.
        h_0 (float, optional): The coefficients of the power spectral density of the clock noise. Defaults to 2e-19.
        h_2 (float, optional): The coefficients of the power spectral density of the clock noise. Defaults to 2e-20.

    Returns:
        np.ndarray: The process noise covariance matrix.
    """
    Q = np.zeros((8, 8))

    # Add the process noise for each coordinate
    Q[:2, :2] = dynamic_position_process_noise_profile(S_x=S_x, dt=dt)
    Q[2:4, 2:4] = dynamic_position_process_noise_profile(S_x=S_y, dt=dt)
    Q[4:6, 4:6] = dynamic_position_process_noise_profile(S_x=S_z, dt=dt)

    # Add the process noise for the clock bias and clock drift
    Q[6:, 6:] = clock_process_noise_profile(dt=dt, h_0=h_0, h_2=h_2)

    return Q
