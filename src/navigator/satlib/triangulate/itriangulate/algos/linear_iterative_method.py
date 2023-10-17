"""This module contains the implementation of the linear iterative method for GPS triangulation.

Functions:
    least_squares(pseudorange : np.array, sv_pos : np.array , weight : np.array = None ,  eps : float = 1e-5) -> np.array:
        Compute the least squares solution to the GPS problem.
        
    _design_matrix(guess : np.array , pseudorange : np.array, sv_pos : np.array) -> np.array:
        Generate a design matrix for the least squares problem. Uses the 1 in the column for the clock offset. The clock offset must be given in terms of the distance it would travel.
"""

import numpy as np

__all__ = ["least_squares"]


def _design_matrix(
    guess: np.array, pseudorange: np.array, sv_pos: np.array
) -> np.array:
    """Generate a design matrix for the least squares problem. Uses the 1 in the column for the clock offset. The clock offset must be given in terms of the distance it would travel.

    Args:
        guess (np.array): The initial guess for the receiver position must be of [x,y,z].T shape.
        pseudorange (np.array):  The pseudorange measurements of shape (num_svs, 1)
        sv_pos (np.array): The satellite positions of shape (num_svs, 3)

    Returns:
        np.array: The design matrix of shape (num_svs, 4)
    """
    if guess.shape != (3, 1):
        raise ValueError("Guess must be a 3x1 vector")

    if pseudorange.shape != (sv_pos.shape[0], 1):
        raise ValueError("Pseudorange must be a nx1 vector")

    if sv_pos.shape[1] != 3:
        raise ValueError("SV Position must be a nx3 vector")

    # Calculate the distance from the guess to the satellite
    P_o = np.linalg.norm(sv_pos - guess.T, axis=1).reshape(-1, 1)

    # Calculate the design matrix
    A = np.hstack([-(sv_pos - guess.T) / P_o, np.ones((sv_pos.shape[0], 1))])

    # Calculate the residual
    r = pseudorange - P_o

    return r, A


def least_squares(
    pseudorange: np.array, sv_pos: np.array, weight: np.array = None, eps: float = 1e-5
) -> np.array:
    """Compute the least squares solution to the GPS problem.

    Args:
        pseudorange (np.array): The pseudorange measurements of shape (num_svs, 1)
        sv_pos (np.array): The satellite positions of shape (num_svs, 3)
        weight (np.array, optional): The weight matrix of shape (num_svs, num_svs). Defaults to None.
        eps (float): The convergence threshold for the iterative solver. Defaults to 1e-10.

    Returns:
        np.array: The computed receiver position of shape (3, 1)
    """
    # Check that the inputs are the correct shape
    if pseudorange.shape != (sv_pos.shape[0], 1):
        raise ValueError("Pseudorange must be a nx1 vector")

    # Check that the inputs are the correct shape
    if sv_pos.shape[1] != 3:
        raise ValueError("SV Position must be a nx3 vector")

    # Initialize weight matrix if not given
    if weight is None:
        weight = np.eye(sv_pos.shape[0]).astype(np.float64)

    # Check that the weight matrix is the correct shape
    if weight.shape != (sv_pos.shape[0], sv_pos.shape[0]):
        raise ValueError("Weight matrix must be a nxn matrix")

    # Convert to np.float64
    pseudorange, sv_pos = pseudorange.astype(np.float64), sv_pos.astype(np.float64)

    # Initialize the guess for the receiver position and clock offset
    guess = np.array([0, 0, 0, 0]).reshape(-1, 1).astype(np.float64)

    # Iterate until convergence
    while True:
        # Generate the design matrix and residual.
        # Pass the coordinates only of the guess since the clock offset is not used in the design matrix
        r, A = _design_matrix(guess[:3, 0].reshape(-1, 1), pseudorange, sv_pos)

        # Solve for the correction
        dr = np.linalg.inv(A.T @ weight @ A) @ A.T @ weight @ r

        # Update the guess
        guess += dr

        # Check for convergence
        if np.linalg.norm(dr[:3, 0]) < eps:
            break

    # Normalize the clock offset
    guess[3, 0] = guess[3, 0] / 299792458

    # Compute the Q matrix
    Q = np.linalg.inv(A.T @ weight @ A)

    # Compute the GDOP, PDOP and TDOP
    dic = {
        "GDOP": np.sqrt(np.trace(Q)),
        "PDOP": np.sqrt(np.trace(Q[:3, :3])),
        "TDOP": np.sqrt(Q[3, 3]),
    }

    return dic, guess.flatten()
