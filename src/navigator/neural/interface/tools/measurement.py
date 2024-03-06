"""The measurement model for Kalman Net Neural Network for GNSS applications.

State Definition:
    The state vector (curr_state) represents the receiver and is defined as follows:
    
    x = [Dx, x_dot, Dy, y_dot, z, z_dot, Dclock_drift]

    Where:
    - Dx : Error in X coordinate from baseline coordinate.
    - x_dot : Velocity of the x-coordinate.
    - Dy : Error in Y coordinate from baseline coordinate.
    - y_dot : Velocity of the y-coordinate.
    - Dz : Error in Y coordinate from baseline coordinate.
    - z_dot : Velocity of the z-coordinate.
    - Dclock_drift: Error in clock drift from baseline coordinate.

Note:
    This measurement model is used in the Neural Kalman Filter for GNSS applications. The error and drift modeling is left to the neural network.
"""

import torch

__all__ = ["kalman_net_measurement_model"]


def kalman_net_measurement_model(
    state: torch.Tensor,
    base_line: torch.Tensor,
    sv_matrix: torch.Tensor,
) -> torch.Tensor:
    """Measurement Function for the Kalman Net Neural Network.

    Args:
        state: The current state of the receiver. [Dx, x_dot, Dy, y_dot, Dz, z_dot, Dclock_drift]
        base_line : The base line initilization for the location in ECEF coordinate. [x_0, x_dot_0, y_0, y_dot_0, z_0, z_dot_0, clock_drift_0]
        sv_matrix: The satellite frame containing the satellite positions, elevation, and azimuth.

    Note:
        The base line is used as the Error State initialization for the Kalman Net Neural Network.
        This is done to avoid the numerical instability of the Neural Network since the state
        coordinate are tracked in the ECEF frame whose order is in 1e6. So, error state is tracked.

    Returns:
        Returns: The pseudo-range measurement from the receiver to the satellites.
    """
    # Add the base line to the error state
    Dposition = state[[0, 2, 4]]
    position = base_line[[0, 2, 4]] + Dposition

    # Compute the range from the user to the satellites
    range_from_user = torch.norm(sv_matrix - position, dim=1)

    # Get the current range from the code measurements
    return range_from_user + state[6]


# name: src/navigator/core/triangulate/itriangulate/kalman/tools/phase_based/measurement.py
