"""Optimized Functional Interface for Extended Kalman Filter (EKF).

This module provides an optimized functional interface for the Extended Kalman Filter (EKF), leveraging Numba for enhanced computation speed during both the predict and update steps.

Functions:
- `ekf_predict_covariance_update(F, P_prior, Q)`: Predict step of the EKF that updates the state covariance matrix.
- `ekf_update(y_hat, x_prior, H, P_prior, R)`: Update step of the EKF.

All functions are implemented with Numba's JIT compilation for improved performance and parallel execution.

Note:
    The functions are meticulously optimized for performance using Numba's Just-In-Time (JIT) compilation and parallelization features, ensuring efficient execution in numerical computing tasks. 
    These functions are not intended for further modification or optimization using Numba, as their design already maximizes computational efficiency.

"""

import numba as nb
import numpy as np

__all__ = [
    "ekf_predict_covariance_update",
    "ekf_update",
]


@nb.njit(
    nb.float64[:, :](nb.float64[:, :], nb.float64[:, :], nb.float64[:, :]),
    fastmath=True,
    error_model="numpy",
    parallel=True,
    cache=True,
)
def ekf_predict_covariance_update(
    F: np.ndarray,
    P_posterior: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """Predict step of the Extended Kalman Filter (EKF) that updates the state covariance matrix.

    Args:
        F (np.ndarray): State transition matrix.
        P_posterior (np.ndarray): The posterior state covariance matrix.
        Q (np.ndarray): Process noise covariance matrix.

    Returns:
        np.ndarray: Updated state covariance matrix.
    """
    return F @ P_posterior @ F.T + Q


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float64[:, :]))(
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
    ),
    fastmath=True,
    error_model="numpy",
    parallel=True,
    cache=True,
)
def ekf_update(
    y_hat: np.ndarray,
    x_prior: np.ndarray,
    H: np.ndarray,
    P_prior: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Update step of the Extended Kalman Filter (EKF).

    Args:
        y_hat (np.ndarray): Predicted measurement.
        x_prior (np.ndarray): Prior state vector.
        H (np.ndarray): Measurement matrix.
        P_prior (np.ndarray): Prior state covariance matrix.
        R (np.ndarray): Measurement noise covariance matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: Updated state vector and state covariance matrix.
    """
    # Innovation covariance
    S = H @ P_prior @ H.T + R
    # Compute the Kalman gain
    K = P_prior @ H.T @ np.linalg.inv(S)
    # Update the state vector
    x = x_prior + K @ y_hat
    # Update the state covariance matrix
    P = (np.eye(K.shape[0]) - K @ H) @ P_prior

    return x, P


@nb.njit(
    nb.types.Tuple((nb.float64[:, :], nb.float64[:, :]))(
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
    ),
    fastmath=True,
    error_model="numpy",
    parallel=True,
    cache=True,
)
def kalman_gain(
    P_prior: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Kalman gain for the Extended Kalman Filter (EKF).

    Args:
        P_prior (np.ndarray): Prior state covariance matrix.
        H (np.ndarray): Measurement matrix.
        R (np.ndarray): Measurement noise covariance matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: Kalman gain and innovation covariance.
    """
    # Innovation covariance
    S = H @ P_prior @ H.T + R
    # Compute the Kalman gain
    K = P_prior @ H.T @ np.linalg.inv(S)

    return K, S
