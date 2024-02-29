"""Phase Based Measurement Model for Kalman Net.

This module defines a measurement function for Phase Based Measurement Model for Kalman Net.


State Definition:
    The state vector (curr_state) represents the receiver and is defined as follows:
    
    x = [Dx, x_dot, Dy, y_dot, z, z_dot, Dclock_drift, clock_drift_rate, wet_tropospheric_delay, B1, ..., Bn]

    Where:
    - Dx : Error in X coordinate from baseline coordinate.
    - x_dot : Velocity of the x-coordinate.
    - Dy : Error in Y coordinate from baseline coordinate.
    - y_dot : Velocity of the y-coordinate.
    - Dz : Error in Y coordinate from baseline coordinate.
    - z_dot : Velocity of the z-coordinate.
    - Dclock_drift: Error in clock drift from baseline coordinate.
    - clock_drift_rate: Clock drift rate.
    - wet_tropospheric_delay: Wet tropospheric delay.
    - B: Bias of the phase measurements, including integer ambiguity and hardware delay.

See Also:
    - Linear observation model for PPP:
      https://gssc.esa.int/navipedia/index.php?title=Linear_observation_model_for_PPP

"""

import torch

__all__ = ["phase_measurement_model"]


def phase_measurement_model(
    error_state: torch.Tensor,
    base_line: torch.Tensor,
    sv_matrix: torch.Tensor,
    num_sv: int,
) -> torch.Tensor:
    """Measurement Function for the phase based GPS Kalman Filter.

    Args:
        base_line : The base line initilization for the location in ECEF coordinate. [x_0 , y_0 , z_0 , cdt_0]
        error_state: The current state of the system.
        sv_matrix: The satellite frame containing the satellite positions, elevation, and azimuth.
        num_sv: The number of satellites to continuously track.

    Returns:
        Returns: The stacked [code, phase] measurements. The code is first, and the phase is second.
    """
    # Add the base line to the error state
    Dposition = error_state[[0, 2, 4]]
    position = base_line[[0, 1, 2]] + Dposition

    # Get the current state variables
    cdt = base_line[[3]] + error_state[[6]]
    t_wet = error_state[[8]]  # t_wet
    integer_bais_vector = error_state[9 : 9 + num_sv]  # lambda_N

    # Compute the range from the user to the satellites
    range_from_user = torch.norm(sv_matrix - position, dim=1)

    # Get the current range from the code measurements
    phase_measurements = (
        range_from_user + cdt + t_wet + integer_bais_vector
    )  # phase_measurements
    code_measurements = range_from_user + cdt + t_wet

    # Return the measurements
    return torch.concatenate((code_measurements, phase_measurements))


# name: src/navigator/core/triangulate/itriangulate/kalman/tools/phase_based/measurement.py
