"""Defines the Measurement Function for the PPP modes.

This model is only defined for the GPS PPP model. This is not multi-GNSS model supported.


StateDefinition:
    curr_state = [x, x_dot , y,  y_dot , z, z_dot , cdt, cdt_dot, t_wet, lambda_N1, ... , lambda_Nn, delta_code1, ... , delta_coden, delta_phase1, ... , delta_phasen]
    
    where:
        x, y, z: are the ECEF coordinates of the receiver.
        x_dot, y_dot, z_dot: are the velocities of the receiver.
        t_wet: is the wet tropospheric delay.
        lambda_N: is the integer ambiguity for the phase measurements.
"""

import numba as nb
import numpy as np

__all__ = ["ppp_measurement_model"]


@nb.njit(
    "float64[:](float64[:], float64[:,:], int64)",
    fastmath=True,
    error_model="numpy",
    nogil=True,
    cache=True,
)
def ppp_measurement_model(
    state: np.ndarray, sv_matrix: np.ndarray, num_sv: int
) -> np.ndarray:
    """Measurement Function for the PPP modes with Kalman Filter.

    Args:
        state: The current state of the system.
        sv_matrix: The ECEF coordinates of the satellites.
        num_sv: The number of satellites to continuously track.

    StateDefinition:
        curr_state = [x, x_dot , y,  y_dot , z, z_dot , cdt, cdt_dot, t_wet, lambda_N1, ... , lambda_Nn]
        where:
            x, y, z: are the ECEF coordinates of the receiver.
            x_dot, y_dot, z_dot: are the velocities of the receiver.
            t_wet: is the wet tropospheric delay.
            lambda_N: is the integer ambiguity for the phase measurements.

    Returns:
        Returns: The stacked [code, phase] measurements. The code is first and the phase is second.
    """
    # Get position from the current state
    position = state[0:5:2]  # x, y, z
    cdt = state[6]  # cdt
    tropo = state[8]  # t_wet
    integer_bais_vector = state[9 : 9 + num_sv]  # lambda_N

    # Calculate the lat, long, height of the reciever

    # Get the current range from the phase measurements
    range_from_user = np.power((sv_matrix - position), 2).sum(axis=1) ** 0.5

    # Get the current range from the code measurements
    phase_measurements = (
        range_from_user + cdt + tropo + integer_bais_vector
    )  # phase_measurements
    code_measurements = range_from_user + cdt + tropo

    # Return the measurements
    return np.concatenate((code_measurements, phase_measurements))
