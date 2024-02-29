"""Defines the Process Noise Profile for a phase-based GPS Kalman Filter.

This module provides functionality for modeling the process noise in a Kalman Filter
for Phase-based Positioning using GPS measurements. It includes the definition of
the state vector and the implementation of the process noise profile.

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

import numpy as np

from ...code_based.noise_profile import (
    clock_process_noise_profile,
    dynamic_position_process_noise_profile,
)

__all__ = ["phase_process_noise_profile"]


def phase_process_noise_profile(
    dt: float,
    num_sv: int,
    S_lat: float = 1e-5,
    S_lon: float = 1e-5,
    S_h: float = 10,
    S_wet: float = 0.1,
    S_b: float = 100,
    h_0: float = 2e-21,
    h_2: float = 2e-23,
) -> np.ndarray:
    """Process Noise Profile for the phase-based GPS Kalman Filter.

    Args:
        dt (float): The time step.
        num_sv (int): The number of satellites to continuously track.
        S_lat (float, optional): Power spectral density for the latitude random walk. Defaults to 1e-3.
        S_lon (float, optional): Power spectral density for the longitude random walk. Defaults to 1e-3.
        S_h (float, optional): Power spectral density for the height random walk. Defaults to 2.
        S_wet (float, optional): Power spectral density for the wet tropospheric delay. Defaults to 0.1.
        S_b (float, optional): Power spectral density for the phase bias. Defaults to 100.
        h_0 (float, optional): The first coefficient of the clock model. Defaults to 2e-21.
        h_2 (float, optional): The second coefficient of the clock model. Defaults to 2e-23.

    Returns:
        np.ndarray: The process noise profile for the current state.
    """
    # Initialize the process noise profile
    Q = np.zeros((9 + num_sv, 9 + num_sv), dtype=np.float64)

    # Position noise profile
    Q[0:2, 0:2] = dynamic_position_process_noise_profile(S_lat, dt)
    Q[2:4, 2:4] = dynamic_position_process_noise_profile(S_lon, dt)
    Q[4:6, 4:6] = dynamic_position_process_noise_profile(S_h, dt)

    # Add clock noise profile
    Q[6:8, 6:8] = clock_process_noise_profile(dt, h_0=h_0, h_2=h_2)

    # Add wet tropospheric delay noise profile
    Q[8, 8] = S_wet * dt

    # Add phase bias noise profile
    Q[9:, 9:] = S_b * dt * np.eye(num_sv, dtype=np.float64)

    return Q
