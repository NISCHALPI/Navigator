"""This module contains the variational loss function for the EKF model described in the paper below.

- https://arxiv.org/pdf/2210.04165
"""

import torch

from .negative_log_likelihood import gaussain_log_likelihood

__all__ = ["log_likelihood_of_observation_model", "kl_divergence_of_transistion_model"]


def log_likelihood_of_observation_model(
    y: torch.Tensor,
    Dh: callable,
    DH_args: tuple,
    x_smoothed: torch.Tensor,
    P_smoothed: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """Computes the log likelihood of the observation model.

    Args:
        y (torch.Tensor) : The observation tensor. (T, dim_y)
        Dh (callable) : The joint jacobian of the observation model.
        DH_args (tuple) : The arguments of the joint jacobian.
        x_smoothed (torch.Tensor) : The smoothed state tensor. (T, dim_x)
        P_smoothed (torch.Tensor) : The smoothed covariance tensor. (T, dim_x, dim_x)
        R (torch.Tensor) : The observation noise covariance tensor. (dim_y, dim_y)

    Returns:
        torch.Tensor
            The log likelihood of the observation model.

    Note:
        - The arguments of the joint jacobian are passed as a tuple whose entries
            must have the temporal dimension as the first dimension.
    """
    # Loop and compute the loss
    loss = 0

    for i in range(y.shape[0]):
        # Compute the joint jacobian and mu
        C, y_mu = Dh(x_smoothed[i], *(args[i] for args in DH_args))

        # Compute the log likelihood
        y_cov = C @ P_smoothed[i] @ C.T + R
        loss += gaussain_log_likelihood(
            innovation=y[i] - y_mu, innovation_covariance=y_cov
        )

    return loss


def kulback_leibler_divergence_gaussian(
    mu_p: torch.Tensor, cov_p: torch.Tensor, mu_q: torch.Tensor, cov_q: torch.Tensor
) -> torch.Tensor:
    """Computes the KL divergence between two Gaussian distributions.

    Args:
        mu_p (torch.Tensor) : The mean of the first Gaussian distribution. (dim_x,)
        cov_p (torch.Tensor) : The covariance of the first Gaussian distribution. (dim_x, dim_x)
        mu_q (torch.Tensor) : The mean of the second Gaussian distribution. (dim_x,)
        cov_q (torch.Tensor) : The covariance of the second Gaussian distribution. (dim_x, dim_x)

    Returns:
        torch.Tensor
            The KL divergence between the two Gaussian distributions.
    """
    # Compute the determinant of the covariances
    det_cov_p = torch.det(cov_p)
    det_cov_q = torch.det(cov_q)

    # Compute the inverse of the covariance
    inv_cov_q = torch.inverse(cov_q)

    # Compute the trace term
    trace_term = torch.trace(inv_cov_q @ cov_p)

    # Compute the difference in the means
    diff_mu = mu_p - mu_q

    # Compute the KL divergence
    return 0.5 * (
        torch.log(det_cov_q / det_cov_p)
        - cov_p.shape[0]
        + trace_term
        + diff_mu @ inv_cov_q @ diff_mu
    )


def kl_divergence_of_transistion_model(
    Df: callable,
    DF_args: tuple,
    x0: torch.Tensor,
    P0: torch.Tensor,
    x_smoothed: torch.Tensor,
    P_smoothed: torch.Tensor,
    Q: torch.Tensor,
) -> torch.Tensor:
    """Computes the KL divergence of the transition model and the posterior.

    Args:
        Df (callable) : The joint jacobian function of the transition model.
        DF_args (tuple) : The arguments of the joint jacobian.
        x0 (torch.Tensor) : The initial smoothed state tensor. (dim_x,)
        P0 (torch.Tensor) : The initial smoodthed covariance tensor. (dim_x, dim_x)
        x_smoothed (torch.Tensor) : The smoothed state tensor. (T, dim_x)
        P_smoothed (torch.Tensor) : The smoothed covariance tensor. (T, dim_x, dim_x)
        Q (torch.Tensor) : The transition noise covariance tensor. (dim_x, dim_x)
    """
    loss = 0

    # Compute the KL divergence
    for i in range(x_smoothed.shape[0]):
        # Compute the joint jacobian and mu
        A, x_mu = Df(x_smoothed[i - 1] if i > 0 else x0, *(arg[i] for arg in DF_args))

        # Propagate the covariance
        P_mu = A @ (P_smoothed[i - 1] if i > 0 else P0) @ A.T + Q

        # Compute the KL divergence
        loss += kulback_leibler_divergence_gaussian(
            mu_p=x_smoothed[i],
            cov_p=P_smoothed[i],
            mu_q=x_mu,
            cov_q=P_mu,
        )

    return loss
