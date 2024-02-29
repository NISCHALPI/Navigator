"""Defines the Process Noise Profile for a phase-based GPS Kalman Filter.

The state vector is parameterized in the cartesian frame instead of the ellipsoidal frame in 
this module.

This module provides functionality for modeling the process noise in a Kalman Filter
for Phase-based Positioning using GPS measurements. It includes the definition of
the state vector and the implementation of the process noise profile.


State Definition:
    The state vector (curr_state) represents the receiver and is defined as follows:
    
    x = [x, x_dot, y, y_dot, z, z_dot, clock_drift, clock_drift_rate, wet_tropospheric_delay, B1, ..., Bn]

    Where:
    - x : x-coordinate of the receiver.
    - x_dot : Velocity of the x-coordinate.
    - y : y-coordinate of the receiver.
    - y_dot : Velocity of the y-coordinate.
    - z : z-coordinate of the receiver.
    - z_dot : Velocity of the z-coordinate.
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
    S_x: float = 0.2,
    S_y: float = 0.2,
    S_z: float = 0.2,
    S_wet: float = 0.1,
    S_b: float = 10,
    h_0: float = 2e-21,
    h_2: float = 2e-23,
) -> np.ndarray:
    """Process Noise Profile for the phase-based GPS Kalman Filter.

    Args:
        dt (float): The time step.
        num_sv (int): The number of satellites to continuously track.
        S_x (float, optional): Power spectral density for the x-coordinate. Defaults to 0.2.
        S_y (float, optional): Power spectral density for the y-coordinate. Defaults to 0.2.
        S_z (float, optional): Power spectral density for the z-coordinate. Defaults to 0.2.
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
    Q[0:2, 0:2] = dynamic_position_process_noise_profile(S_x=S_x, dt=dt)
    Q[2:4, 2:4] = dynamic_position_process_noise_profile(S_x=S_y, dt=dt)
    Q[4:6, 4:6] = dynamic_position_process_noise_profile(S_x=S_z, dt=dt)

    # Add clock noise profile
    Q[6:8, 6:8] = clock_process_noise_profile(dt, h_0=h_0, h_2=h_2)

    # Add wet tropospheric delay noise profile
    Q[8, 8] = S_wet * dt

    # Add phase bias noise profile
    Q[9:, 9:] = S_b * dt * np.eye(num_sv, dtype=np.float64)

    return Q
