"""This module contains the noise profile for the Kalman Filter based triangulation methods.

Noise profile:
    - 

"""

import numpy as np

__all__ = ["clock_process_noise_profile", "dynamic_position_process_noise_profile"]


# Speed of light
SPEED_OF_LIGHT = 299792458  # m/s


# Typical Power Spectral Density (PSD) of the clock noise
PSD = {
    "compensated_crystal": {"h_0": 2e-19, "h_1": 7e-21, "h_2": 2e-20},
    "ovenized_crystal": {"h_0": 8e-20, "h_1": 2e-21, "h_2": 4e-23},
    "rubidium": {"h_0": 2e-20, "h_1": 7e-24, "h_2": 4e-29},
}


def clock_process_noise_profile(
    dt: float,
    h_0: float = 2e-19,
    h_2: float = 2e-20,
) -> np.ndarray:
    """This function returns the clock noise profile.

    Args:
        h_0 (float): Coefficients of the power spectral density of the clock noise.
        h_2 (float): Coefficients of the power spectral density of the clock noise.
        dt (float): Time step in seconds.

    Returns:
        np.ndarray: Clock noise profile 2x2 matrix.
    """
    # Caclulate the clock noise profile
    S_g = 2 * np.pi**2 * h_2 * SPEED_OF_LIGHT**2  # Ensure that the units are meters
    S_f = (h_0 / 2) * SPEED_OF_LIGHT**2  # Ensure that the units are meters

    return np.array(
        [[S_f + (S_g * dt**2 / 3), S_g * dt**2 / 2], [S_g * dt**2 / 2, S_g * dt]],
        dtype=np.float64,
    )


def dynamic_position_process_noise_profile(
    S_x: float,
    dt: float,
) -> np.ndarray:
    """This function returns the position noise profile for constant velocity model.

    Args:
        S_x (float): Spectral amplitude.
        dt (float): Time step in seconds.

    Returns:
        np.ndarray: Position noise profile 2x2 matrix.
    """
    return np.array(
        [[S_x * dt**3 / 3, S_x * dt**2 / 2], [S_x * dt**2 / 2, S_x * dt]],
        dtype=np.float64,
    )


def diagonal_measurement_noise_profile(
    sigma_r: float,
    n_measurement: int,
) -> np.ndarray:
    """This function returns the diagonal measurement noise profile.

    Args:
        sigma_r (float): Measurement noise standard deviation.
        n_measurement (int): Number of measurements.


    Returns:
        np.ndarray: Measurement noise profile 2x2 matrix.
    """
    return np.eye(n_measurement, dtype=np.float64) * sigma_r**2
