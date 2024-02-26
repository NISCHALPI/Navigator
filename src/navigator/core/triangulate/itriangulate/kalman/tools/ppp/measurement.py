"""Defines the Measurement Function for the PPP modes.

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

__all__ = ["ppp_measurement_model"]


def ppp_measurement_model(state: np.ndarray, sv_matrix: np.ndarray) -> np.ndarray:
    """Measurement Function for the PPP modes with Kalman Filter.

    Args:
        state: The current state of the system.
        sv_matrix: The ECEF coordinates of the satellites.

    StateDefinition:
        curr_state = [x, x_dot , y,  y_dot , z, z_dot , cdt, cdt_dot, t_wet, lambda_N, delta_code , delta_phase]
        where:
            x, y, z: are the ECEF coordinates of the receiver.
            x_dot, y_dot, z_dot: are the velocities of the receiver.
            t_wet: is the wet tropospheric delay.
            lambda_N: is the integer ambiguity for the phase measurements.

    Returns:
        Returns: The stacked [code, phase] measurements. The code is first and the phase is second.
    """
    # Get position from the current state
    position = state[[0, 2, 4]]  # x, y, z
    cdt = state[[6]]  # cdt
    tropo = state[[8]]  # t_wet
    integer_bais = state[[9]]  # lambda_N
    code_bias = state[[10]]  # code specific bias
    phase_bias = state[[11]]  # phase specific bias

    # Get the current range from the phase measurements
    range_from_user = np.linalg.norm(sv_matrix - position, axis=1)

    # Get the current range from the code measurements
    phase_measurements = range_from_user + cdt + tropo + integer_bais + phase_bias
    code_measurements = range_from_user + cdt + tropo + code_bias

    # Return the measurements
    return np.concatenate((code_measurements, phase_measurements))
