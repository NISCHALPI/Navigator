"""The state transistion model for Kalman Net Neural Network for GNSS applications.

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
    This state transistion model is used in the Neural Kalman Filter for GNSS applications. The error and drift modeling is left to the neural network.
"""

import torch

__all__ = ["kalman_net_state_transistion_model"]


def kalman_net_state_transistion_model(
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

    # State transition matrix 7x7
    F = torch.eye(7, dtype=torch.float32)
    F[:2, :2] = A
    F[2:4, 2:4] = A
    F[4:6, 4:6] = A

    return F
