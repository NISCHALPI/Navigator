"""Defines the State Transition Function for a error state phase-based GPS Kalman Net.

The state vector is parameterized in the cartesian frame which predicts the error in the consecutive
measurements instead of the predicting the whole coordinate which helps in the numerical stability as 
well as training.


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

"""

import torch


def phase_state_transistion_matrix(
    dt: float,
    # num_sv: int,
) -> torch.Tensor:
    """State Transition Matrix for the phase-based GPS Kalman Filter.

    Args:
        dt (float): The time step.
        num_sv (int): The number of satellites to continuously track.

    Returns:
        torch.Tensor : The state transition matrix for the current state.
    """
    # Constant velocity state transition matrix
    A = torch.eye(2, dtype=torch.float32)
    A[0, 1] = dt

    F = torch.eye(9, dtype=torch.float32)
    F_xyz = torch.kron(torch.eye(4), A)

    # Set the state transition matrix for the ellipsoidal coordinates, clock drift
    F[:8, :8] = F_xyz
    # Other parameters are independent of the state and time hence the state transition matrix is an identity matrix

    return F
