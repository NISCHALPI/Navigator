"""Utility functions for calculating negative log likelihood of observations.

Functions:
    - gaussain_log_likelihood: Calculates the likelihood estimation at a single time step.
    - log_likelihood: Calculates the log likelihood estimation for a sequence of time steps.
    - negative_log_likelihood: Calculates the negative log likelihood estimation for a sequence of time steps.

The negative log likelihood for a sequence of innovations and their covariances is computed using the 
logarithm of the joint distribution of the observations given the parameters.

"""

import torch

__all__ = ["gaussain_log_likelihood", "log_likelihood", "negative_log_likelihood"]


def gaussain_log_likelihood(
    innovation: torch.Tensor,
    innovation_covariance: torch.Tensor,
) -> torch.Tensor:
    """Calculates the likelihood estimation at a single time step.

    Args:
        innovation (torch.Tensor): The innovation vector, representing the difference
                                   between the observed and predicted measurements.
                                   Shape: (dim_z,)
        innovation_covariance (torch.Tensor): The innovation covariance matrix,
                                              representing the uncertainty in the innovation.
                                              Shape: (dim_z, dim_z)

    Returns:
        torch.Tensor: The loss value for the given innovation and its covariance.

    Note:
        - The dimension factor is skipped since it is a constant and does not affect the optimization.

    Example:
        >>> innovation = torch.tensor([1.0, 2.0])
        >>> innovation_covariance = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> loss = likelihood_at_time(innovation, innovation_covariance)
        >>> print(loss)
    """
    # Calculate the log determinant of the innovation covariance matrix
    log_det = torch.linalg.slogdet(innovation_covariance)[1]

    return -0.5 * (
        log_det + innovation @ torch.linalg.inv(innovation_covariance) @ innovation
    )


def log_likelihood(
    innovation: torch.Tensor,
    innovation_covariance: torch.Tensor,
) -> torch.Tensor:
    """Calculates the log likelihood estimation for a sequence of time steps.

    Args:
        innovation (torch.Tensor): The innovation vector for each time step,
                                   representing the differences between observed and predicted
                                   measurements over a sequence. Shape: (seq_len, dim_z)
        innovation_covariance (torch.Tensor): The innovation covariance matrix for each time step,
                                              representing the uncertainty in the innovations
                                              over a sequence. Shape: (seq_len, dim_z, dim_z)

    Returns:
        torch.Tensor: The total log likelihood value for the sequence of innovations and
                      their covariances.

    Example:
        >>> innovation = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
        >>> innovation_covariance = torch.tensor([
        >>>     [[1.0, 0.0], [0.0, 1.0]],
        >>>     [[0.5, 0.0], [0.0, 0.5]]
        >>> ])
        >>> total_log_likelihood = log_likelihood(innovation, innovation_covariance)
        >>> print(total_log_likelihood)
    """
    return torch.vmap(gaussain_log_likelihood, in_dims=0)(
        innovation, innovation_covariance
    ).sum()


def negative_log_likelihood(
    innovation: torch.Tensor,
    innovation_covariance: torch.Tensor,
) -> torch.Tensor:
    """Calculates the negative log likelihood estimation for a sequence of time steps.

    Args:
        innovation (torch.Tensor): The innovation vector for each time step,
                                   representing the differences between observed and predicted
                                   measurements over a sequence. Shape: (seq_len, dim_z)
        innovation_covariance (torch.Tensor): The innovation covariance matrix for each time step,
                                              representing the uncertainty in the innovations
                                              over a sequence. Shape: (seq_len, dim_z, dim_z)

    Returns:
        torch.Tensor: The total negative log likelihood value for the sequence of innovations and
                      their covariances.

    Example:
        >>> innovation = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
        >>> innovation_covariance = torch.tensor([
        >>>     [[1.0, 0.0], [0.0, 1.0]],
        >>>     [[0.5, 0.0], [0.0, 0.5]]
        >>> ])
        >>> total_negative_log_likelihood = negative_log_likelihood(innovation, innovation_covariance)
        >>> print(total_negative_log_likelihood)
    """
    return -log_likelihood(innovation, innovation_covariance)
