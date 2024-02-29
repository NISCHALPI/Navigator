"""This module contains the implementation of the linear iterative method for GPS triangulation.

Functions:
    least_squares(pseudorange : np.array, sv_pos : np.array , weight : np.array = None ,  eps : float = 1e-5) -> np.array:
        Compute the least squares solution to the GPS problem.
        
    _design_matrix(guess : np.array , pseudorange : np.array, sv_pos : np.array) -> np.array:
        Generate a design matrix for the least squares problem. Uses the 1 in the column for the clock offset. The clock offset must be given in terms of the distance it would travel.
"""

import warnings

import numpy as np
from numba import float64, njit
from numba.core.errors import NumbaPerformanceWarning
from numba.types import UniTuple

# Filter out the numba warnings for this module
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

__all__ = ["least_squares"]


@njit(
    UniTuple(float64[:, :], 2)(float64[:, :], float64[:, :], float64[:, :]),
    fastmath=True,
    parallel=True,
    cache=True,
)
def _design_matrix(
    guess: np.ndarray, pseudorange: np.ndarray, sv_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a design matrix for the least squares problem. Uses the 1 in the column for the clock offset. The clock offset must be given in terms of the distance it would travel.

    Args:
        guess (np.array): The initial guess for the receiver position must be of shape (3, 1).
        pseudorange (np.array):  The pseudorange measurements of shape (num_svs, 1)
        sv_pos (np.array): The satellite positions of shape (num_svs, 3)

    Returns:
        Tuple[np.array, np.array]: The residual and design matrix of shape (num_svs, 1) and (num_svs, 4) respectively.
    """
    if guess.shape != (3, 1):
        raise ValueError("Guess must be a 3x1 vector")

    if pseudorange.shape != (sv_pos.shape[0], 1):
        raise ValueError("Pseudorange must be a nx1 vector")

    if sv_pos.shape[1] != 3:
        raise ValueError("SV Position must be a nx3 vector")

    # Calculate the distance from the guess to the satellite
    P_o = np.sqrt(((sv_pos - guess.T) ** 2).sum(axis=1)).reshape(-1, 1)

    # Calculate the design matrix
    A = np.ones((sv_pos.shape[0], 4), dtype=np.float64)
    A[:, :3] = -(sv_pos - guess.T) / P_o

    # Calculate the residual
    r = pseudorange - P_o

    return r, A


@njit(
    UniTuple(float64[:, :], 3)(
        float64[:, :],
        float64[:, :],
        float64[:, :],
        float64,
    ),
    fastmath=True,
    parallel=True,
    cache=True,
)
def least_squares(
    pseudorange: np.ndarray,
    sv_pos: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the least squares solution to the GPS problem.

    Args:
        pseudorange (np.array): The pseudorange measurements of shape (num_svs, 1)
        sv_pos (np.array): The satellite positions of shape (num_svs, 3)
        weight (np.array, optional): The weight matrix of shape (num_svs, num_svs). Defaults to None.
        eps (float): The convergence threshold for the iterative solver. Defaults to 1e-10.

    Returns:
        Tuple[np.array, np.array, np.array]: The guess (4 x 1) , covariance matrix (4x4), and sigma_o (1x1)
    """
    # Check that the inputs are the correct shape
    if pseudorange.shape != (sv_pos.shape[0], 1):
        raise ValueError("Pseudorange must be a nx1 vector")

    # Check that the inputs are the correct shape
    if sv_pos.shape[1] != 3:
        raise ValueError("SV Position must be a nx3 vector")

    # Check that the weight matrix is the correct shape
    if weight.shape != (sv_pos.shape[0], sv_pos.shape[0]):
        raise ValueError("Weight matrix must be a nxn matrix")

    # # Convert to np.float64
    # pseudorange, sv_pos = pseudorange.astype(np.float64), sv_pos.astype(np.float64)

    # Initialize the guess for the receiver position and clock offset
    guess = np.zeros((4, 1), dtype=np.float64)

    # Add counter to prevent infinite loop
    counter = 0
    # Iterate until convergence
    while True:
        # Generate the design matrix and residual.
        # Pass the coordinates only of the guess since the clock offset is not used in the design matrix
        r, A = _design_matrix(
            guess=guess[:3, 0:],
            pseudorange=pseudorange,
            sv_pos=sv_pos,
        )

        # Solve for the correction
        dr = np.linalg.inv(A.T @ weight @ A) @ A.T @ weight @ r

        # Update the guess
        guess += dr

        # Check for convergence
        if np.linalg.norm(dr[:3, 0]) < eps or counter > 100000:
            break
        # Update the counter
        counter += 1

    # Calculate Variance which is proportional to magnitude of the residual
    sigma_o = np.sqrt(r.T @ weight @ r / (sv_pos.shape[0] - 4))

    # Compute the Q matrix
    Q = np.linalg.inv(A.T @ weight @ A)

    # Calculate the covariance matrix
    cov = sigma_o[0, 0] ** 2 * Q  # Note: Sigma is a 1x1 matrix

    # Return the guess, covariance, and sigma_o
    return (guess, cov, sigma_o)
