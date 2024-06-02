"""Weighted Least Squares algorithms for triangulation problems."""

import numpy as np
from .wls import non_linear_weighted_least_squares


__all__ = ["wls_triangulation"]


def HJacobian(x: np.ndarray, sv_pos: np.ndarray) -> np.ndarray:
    """
    Computes the Jacobian of the observation model for the non-linear least squares problem.

    Args:
        x (np.ndarray): The estimated receiver position and clock bias (cdt) of shape (4,)
        sv_pos (np.ndarray): The satellite positions of shape (num_svs, 3).

    Returns:
        np.ndarray: The Jacobian of the observation model of shape (num_svs, 4).
    """
    num_svs = sv_pos.shape[0]
    H = np.zeros((num_svs, 4))

    # Calculate the estimated ranges
    rho = f(x, sv_pos)  # (num_svs,)

    # Calculate the Jacobian
    H[:, :-1] = (x[:3] - sv_pos) / rho[:, None]
    H[:, -1] = 1

    return H


def f(x: np.ndarray, sv_pos: np.ndarray) -> np.ndarray:
    """
    Computes the non-linear observation model for the least squares problem.

    Args:
        x (np.ndarray): The estimated receiver position and clock bias (cdt) of shape (4,)
        sv_pos (np.ndarray): The satellite positions of shape (num_svs, 3).

    Returns:
        np.ndarray: The non-linear observation model of shape (num_svs, 1).
    """
    return np.linalg.norm(x[:3] - sv_pos, axis=1) + x[3]


def wls_triangulation(
    pseudorange: np.ndarray,
    sv_pos: np.ndarray,
    W: np.ndarray,
    x0: np.ndarray = np.zeros(4),
    max_iter: int = 1000,
    eps: float = 1e-5,
) -> dict[str, np.ndarray | float]:
    """
    Estimates the receiver position and clock bias using the non-linear weighted least squares method given noisy pseudorange measurements and satellite positions.

    Args:
        pseudorange (np.ndarray): The pseudorange measurements of shape (num_svs).
        sv_pos (np.ndarray): The satellite positions of shape (num_svs, 3).
        x0 (np.ndarray): The initial guess for the receiver position and clock bias of shape (4, 1).
        W (np.ndarray): The weight matrix of shape (num_svs, num_svs).
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        eps (float, optional): The convergence threshold. Defaults to 1e-5.

    Returns:
        tuple[np.ndarray, np.ndarray]: The estimated receiver position and clock bias of shape (4, 1) and the error covariance matrix of the residuals of shape (num_svs, num_svs).
    """
    sol, covar = non_linear_weighted_least_squares(
        pseudorange,
        f,
        HJacobian,
        W,
        x0,
        eps=eps,
        max_iter=max_iter,
        f_args=(sv_pos,),
        HJacobian_args=(sv_pos,),
    )

    # Calculate the DOPS for satellites
    H = HJacobian(sol, sv_pos)

    # Calculate the covariance matrix for dops calculation
    Q = H.T @ W @ H

    # Calulate the DOPs values
    gdop = np.sqrt(np.trace(Q))
    pdop = np.sqrt(np.trace(Q[:3, :3]))
    tdop = np.sqrt(Q[3, 3])
    hdop = np.sqrt(Q[0, 0] + Q[1, 1])
    vdop = np.sqrt(Q[2, 2])

    # Calculate the error covariance matrix of the residuals
    return {
        "solution": sol,
        "error_covariance": covar,
        "dops": {
            "gdop": gdop,
            "pdop": pdop,
            "tdop": tdop,
            "hdop": hdop,
            "vdop": vdop,
        },
    }
