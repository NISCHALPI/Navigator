"""Parametric Extended Kalman Filter (EKF) implementation for state estimation in nonlinear dynamic systems.

This module provides a class `PEKF` that implements the Extended Kalman Filter (EKF) algorithm for state estimation in nonlinear dynamic systems.
The EKF algorithm estimates the state of a discrete-time controlled process that is governed by a nonlinear stochastic difference equation.
The parametric variant of the EKF allows for data-driven learning of hyperparameters and model parameters for the state transition and measurement
functions using neural networks and backpropagation algorithms. The loss function for training the PEKF is the Maximum Likelihood Estimation (MLE)
loss which is computed using the joint log-likelihood of measurements.

Classes:
    EKF: Implements the Extended Kalman Filter.

Functions:
    joint_jacobian_transform: Transforms a function(callable or nn.Module) to compute its Jacobian and value in single pass.
    log_likelihood: Computes the log-likelihood for given residuals (innovation) and it's covariance.

Example:
    ```
    # Define state transition and measurement functions as nn.Module
    f = StateTransitionFunction()
    h = MeasurementFunction()

    # Initialize the EKF
    ekf = EKF(dim_x=4, dim_z=2, f=f, h=h)

    # Define initial state, covariance, and noise matrices
    x0 = torch.zeros(4)
    P0 = torch.eye(4)
    Q = torch.eye(4) * 0.1
    R = torch.eye(2) * 0.1

    # Define measurement sequence
    z = torch.rand(10, 2)

    # Perform batch filtering and smoothing
    results = ekf.batch_smoothing(z, x0, P0, Q, R)

    # Calculate the loss for training
    loss = ekf(z, x0, P0, Q, R)
    ```
"""

import torch
import torch.nn as nn

from .joint_jacobian_function import joint_jacobian_transform
from .loss.negative_log_likelihood import log_likelihood
from .loss.varaitional_loss import (
    kl_divergence_of_transistion_model,
    log_likelihood_of_observation_model,
)

__all__ = [
    "ParametricExtendedKalmanFilter",
    "NegativeLogLikelihoodPEKF",
    "VariationalNEKF",
]


class ParametricExtendedKalmanFilter(nn.Module):
    """Implementation of the Extended Kalman Filter (EKF) algorithm for state estimation.

    Args:
        dim_x (int): Dimension of the state vector.
        dim_z (int): Dimension of the measurement vector.
        f (nn.Module): State transition function.
        h (nn.Module): Measurement function.

    Attributes:
        dim_x (int): Dimension of the state vector.
        dim_z (int): Dimension of the measurement vector.
        f (nn.Module): State transition function.
        h (nn.Module): Measurement function.
        I (torch.Tensor): Identity matrix of size (dim_x, dim_x).
        _f (Callable): State transition function with Jacobian computation.
        _h (Callable): Measurement function with Jacobian computation.

    Methods:
        predict: Predicts the state of the system.
        update: Updates the state estimate based on the measurement.
        predict_update: Runs the predict-update loop.
        batch_filtering: Processes the sequence of measurements.
        fixed_interval_smoothing: Performs fixed-interval smoothing on the state estimates.
        batch_smoothing: Processes the sequence of measurements to form a Maximum Likelihood Estimation (MLE) loss.
        autocorreleation: Computes the autocorrelation of the innovation residuals sequence.
        forward: Processes the sequence of measurements to form a Maximum Likelihood Estimation (MLE) loss.
    """

    TERMS = {
        "PriorEstimate": "x_prior",
        "PriorCovariance": "P_prior",
        "StateJacobian": "State_Jacobian",
        "PosteriorEstimate": "x_posterior",
        "PosteriorCovariance": "P_posterior",
        "InnovationResidual": "innovation_residual",
        "InnovationCovariance": "innovation_covariance",
        "KalmanGain": "Kalman_gain",
        "MeasurementJacobian": "Measurement_Jacobian",
        "SmoothedEstimate": "x_smoothed",
        "SmoothedCovariance": "P_smoothed",
        "SmoothedInitialEstimate": "x0_smoothed",
        "SmoothedInitialCovariance": "P0_smoothed",
    }

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        f: nn.Module,
        h: nn.Module,
    ) -> None:
        """Initializes the PEKF algorithm.

        Args:
            dim_x (int): Dimension of the state vector.
            dim_z (int): Dimension of the measurement vector.
            f (nn.Module): State transition function.
            h (nn.Module): Measurement function.

        Note:
            - The state transition function f signature is f(x: torch.Tensor, *args) -> torch.Tensor
            - The measurement function h signature is h(x: torch.Tensor, *args) -> torch.Tensor
            - Any argument must scale with the batch dimension.
        """
        super().__init__()
        # Store the dimensions of the matrices
        self.dim_x = dim_x
        self.dim_z = dim_z

        # Store the state transition and measurement functions
        self.f = f
        self.h = h

        # Register the jacobian functions
        self._f = joint_jacobian_transform(f)
        self._h = joint_jacobian_transform(h)

    def predict(
        self,
        x_posterior: torch.Tensor,
        P_posterior: torch.Tensor,
        Q: torch.Tensor | nn.Parameter,
        f_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """Predicts the state of the system.

        Args:
            x_posterior (torch.Tensor): Posterior state estimate. (dim_x, )
            P_posterior (torch.Tensor): Posterior state error covariance. (dim_x, dim_x)
            Q (torch.Tensor | nn.Parameter): Process noise covariance. (dim_x, dim_x)
            f_args (tuple, optional): Additional arguments for the state transition function. Defaults to ().

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the predicted state estimate, state error covariance, and the state transition matrix.

        Note:
            - f_args must scale with the batch dimension.
        """
        # Return the predicted state and state error covariance
        F, x_prior = self._f(x_posterior, *f_args)

        P_prior = F @ P_posterior @ F.T + Q

        return {
            self.TERMS["PriorEstimate"]: x_prior,
            self.TERMS["PriorCovariance"]: P_prior,
            self.TERMS["StateJacobian"]: F,
        }

    def update(
        self,
        z: torch.Tensor,
        x_prior: torch.Tensor,
        P_prior: torch.Tensor,
        R: torch.Tensor | nn.Parameter,
        h_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """Updates the state estimate based on the measurement.

        Args:
            x_prior (torch.Tensor): Prior state estimate. (dim_x, )
            P_prior (torch.Tensor): Prior state error covariance. (dim_x, dim_x)
            z (torch.Tensor): Measurement vector. (dim_z, )
            R (torch.Tensor | nn.Parameter): Measurement noise covariance. (dim_z, dim_z)
            h_args (tuple, optional): Additional arguments for the measurement function. Defaults to ().

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the updated state estimate, state error covariance, and the innovation.

        Note:
            - h_args must scale with the batch dimension.
        """
        # Compute the predicted measurement and the Jacobian
        H, z_pred = self._h(x_prior, *h_args)

        # Compute the innovation
        y = z - z_pred
        # Compute the innovation covariance matrix
        S = H @ P_prior @ H.T + R
        # Compute the Kalman gain
        K = P_prior @ H.T @ torch.linalg.inv(S)

        # Update the state vector
        x_post = x_prior + K @ y
        # Update the state covariance matrix using joseph form since
        # EKF is not guaranteed to be optimal
        factor = torch.eye(self.dim_x, device=x_post.device, dtype=x_post.dtype) - K @ H
        P_post = factor @ P_prior @ factor.T + K @ R @ K.T

        return {
            self.TERMS["PosteriorEstimate"]: x_post,
            self.TERMS["PosteriorCovariance"]: P_post,
            self.TERMS["InnovationResidual"]: y,
            self.TERMS["InnovationCovariance"]: S,
            self.TERMS["KalmanGain"]: K,
            self.TERMS["MeasurementJacobian"]: H,
        }

    def predict_update(
        self,
        x_posterior: torch.Tensor,
        P_posterior: torch.Tensor,
        z: torch.Tensor,
        Q: torch.Tensor | nn.Parameter,
        R: torch.Tensor | nn.Parameter,
        f_args: tuple = (),
        h_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """Runs the predict-update loop.

        Args:
            x_posterior (torch.Tensor): Posterior state estimate. (dim_x, )
            P_posterior (torch.Tensor): Posterior state error covariance. (dim_x, dim_x)
            z (torch.Tensor): Measurement vector. (dim_z, )
            Q (torch.Tensor | nn.Parameter): Process noise covariance. (dim_x, dim_x)
            R (torch.Tensor | nn.Parameter): Measurement noise covariance. (dim_z, dim_z)
            f_args (tuple, optional): Additional arguments for the state transition function. Defaults to ().
            h_args (tuple, optional): Additional arguments for the measurement function. Defaults to ().

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the state estimates and state error covariances.

        Note:
            - h_args and f_args must scale with the batch dimension.
        """
        # Predict the state
        prediction = self.predict(x_posterior, P_posterior, Q, f_args)

        # Update the state
        update = self.update(
            z=z,
            x_prior=prediction[self.TERMS["PriorEstimate"]],
            P_prior=prediction[self.TERMS["PriorCovariance"]],
            R=R,
            h_args=h_args,
        )

        return {**prediction, **update}

    def batch_filtering(
        self,
        z: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        Q: torch.Tensor | nn.Parameter,
        R: torch.Tensor | nn.Parameter,
        f_args: tuple = (),
        h_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """Processes the sequence of measurements.

        Args:
            z (torch.Tensor): Measurement sequence. (num_timesteps, dim_z)
            x0 (torch.Tensor): Initial state estimate. (dim_x, )
            P0 (torch.Tensor): Initial state error covariance. (dim_x, dim_x)
            Q (torch.Tensor | nn.Parameter): Process noise covariance. (dim_x, dim_x)
            R (torch.Tensor | nn.Parameter): Measurement noise covariance. (dim_z, dim_z)
            f_args (tuple, optional): Additional arguments for the state transition function. Defaults to ().
            h_args (tuple, optional): Additional arguments for the measurement function. Defaults to ().

        Returns:
            dict[str, list[torch.Tensor]]: Dictionary containing lists of state estimates and state error covariances at each time step.

        Note:
            - h_args and f_args must scale with the batch dimension.
        """
        # Sequence length
        T = z.shape[0]
        # Initialize the intermediate variables
        output = {
            self.TERMS[key]: []
            for key in self.TERMS.keys()
            if not key.startswith("Smoothed")
        }

        # Run the filtering algorithm
        for t in range(T):
            # Perform the predict-update loop
            results = self.predict_update(
                x_posterior=(
                    x0 if t == 0 else output[self.TERMS["PosteriorEstimate"]][-1]
                ),
                P_posterior=(
                    P0 if t == 0 else output[self.TERMS["PosteriorCovariance"]][-1]
                ),
                z=z[t],
                Q=Q,
                R=R,
                f_args=(args[t] for args in f_args),
                h_args=(args[t] for args in h_args),
            )

            # Update the output
            for term in output:
                output[term].append(results[term])

        # Stack the results
        for term in output:
            output[term] = torch.stack(output[term])

        return output

    def fixed_interval_smoothing(
        self,
        x0: torch.Tensor,
        P0: torch.Tensor,
        x_posterior: torch.Tensor,
        P_posterior: torch.Tensor,
        FJacobians: torch.Tensor,
        Q: torch.Tensor | nn.Parameter,
    ) -> dict[str, torch.Tensor]:
        """Performs fixed-interval smoothing on the state estimates.

        Args:
            x0 (torch.Tensor): Initial state estimate. (dim_x, )
            P0 (torch.Tensor): Initial state error covariance. (dim_x, dim_x)
            x_posterior (torch.Tensor): Filtered state estimates and covariances. (T, dim_x)
            P_posterior (torch.Tensor): Filtered state error covariances. (T, dim_x, dim_x)
            FJacobians (torch.Tensor): State transition Jacobians. (T, dim_x, dim_x)
            Q (torch.Tensor): Process noise covariance. (dim_x, dim_x)

        Returns:
            dict[str, list[torch.Tensor]]: Dictionary containing lists of smoothed state estimates and state error covariances at each time step.
        """
        # Initialize the smoothed state estimates and state error covariances
        x_smoothed = []
        P_smoothed = []

        # Last state estimate is already the smoothed state estimate
        x_smoothed.append(x_posterior[-1])
        P_smoothed.append(P_posterior[-1])

        # Sequence length
        T = x_posterior.shape[0]

        # Loop and perform fixed-interval smoothing from T - 2 (i.e second last state estimate) to 0 (i.e first state estimate)
        for t in range(T - 2, -1, -1):
            # Compute the prior covariance
            P_prior = FJacobians[t + 1] @ P_posterior[t] @ FJacobians[t + 1].T + Q
            # Compute the smoothing gain
            L = P_posterior[t] @ FJacobians[t + 1].T @ torch.linalg.inv(P_prior)

            # Compute the smoothed state estimate
            x_smoothed.insert(
                0,
                x_posterior[t]
                + L @ (x_smoothed[0] - FJacobians[t + 1] @ x_posterior[t]),
            )
            # Compute the smoothed state error covariance
            P_smoothed.insert(0, P_posterior[t] + L @ (P_smoothed[0] - P_prior) @ L.T)

        # Smoothed initial state estimate and state error covariance
        P_prior = FJacobians[0] @ P0 @ FJacobians[0].T + Q
        L = P0 @ FJacobians[0].T @ torch.linalg.inv(P_prior)
        x0_smoothed = x0 + L @ (x_smoothed[0] - FJacobians[0] @ x0)
        P0_smoothed = P0 + L @ (P_smoothed[0] - P_prior) @ L.T

        return {
            self.TERMS["SmoothedEstimate"]: torch.stack(x_smoothed),
            self.TERMS["SmoothedCovariance"]: torch.stack(P_smoothed),
            self.TERMS["SmoothedInitialEstimate"]: x0_smoothed,
            self.TERMS["SmoothedInitialCovariance"]: P0_smoothed,
        }

    def batch_smoothing(
        self,
        z: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        Q: torch.Tensor | nn.Parameter,
        R: torch.Tensor | nn.Parameter,
        f_args: tuple = (),
        h_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """Processes the sequence of measurements to form an Maximum Likelihood Estimation (MLE) loss.

        Args:
            z (torch.Tensor): Measurement sequence. (T, dim_z)
            x0 (torch.Tensor): Initial state estimate. (dim_x, )
            P0 (torch.Tensor): Initial state covariance. (dim_x, dim_x)
            Q (torch.Tensor | nn.Parameter): Process noise covariance. (dim_x, dim_x)
            R (torch.Tensor | nn.Parameter): Measurement noise covariance. (dim_z, dim_z)
            f_args (tuple, optional): Additional arguments for the state transition function. Defaults to ().
            h_args (tuple, optional): Additional arguments for the measurement function. Defaults to ().

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the state estimates and state error covariances.

        Note:
            - h_args and f_args must scale with the time dimension.
        """
        # Process the measurements
        results = self.batch_filtering(z, x0, P0, Q, R, f_args, h_args)

        # Perform fixed-interval smoothing
        smoothed = self.fixed_interval_smoothing(
            x0,
            P0,
            results[self.TERMS["PosteriorEstimate"]],
            results[self.TERMS["PosteriorCovariance"]],
            results[self.TERMS["StateJacobian"]],
            Q,
        )
        return {**results, **smoothed}

    @staticmethod
    def autocorreleation(
        innovation_residuals: torch.Tensor,
        lag: int = 0,
    ) -> torch.Tensor:
        """Computes the autocorrelation of the innovation residuals sequence.

        Args:
            innovation_residuals (torch.Tensor): Innovation residuals. (T, dim_z)
            lag (int, optional): Lag. Defaults to 1.

        Returns:
            torch.Tensor: Autocorrelation.
        """
        # If the T dimension is less than the lag, return 0
        if innovation_residuals.shape[0] < lag:
            return 0

        # Center the residuals
        residuals = innovation_residuals - torch.mean(innovation_residuals, dim=0)

        # Compute the outer product expectation of the residuals
        outer_product = 0
        for i in range(len(residuals) - lag):
            outer_product += torch.outer(residuals[i], residuals[i + lag])

        # Compute the autocorrelation
        return outer_product / (len(residuals) - lag)


class NegativeLogLikelihoodPEKF(ParametricExtendedKalmanFilter):
    """This class calculates the Maximum Likelihood Estimation (MLE) loss for training the PEKF."""

    def forward(
        self,
        z: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        Q: torch.Tensor | nn.Parameter,
        R: torch.Tensor | nn.Parameter,
        f_args: tuple = (),
        h_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """Processes the sequence of measurements to form an Maximum Likelihood Estimation (MLE) loss.

        Args:
            z (torch.Tensor): Measurement sequence. (T, dim_z)
            x0 (torch.Tensor): Initial state estimate. (dim_x, )
            P0 (torch.Tensor): Initial state covariance. (dim_x, dim_x)
            Q (torch.Tensor | nn.Parameter): Process noise covariance. (dim_x, dim_x)
            R (torch.Tensor | nn.Parameter): Measurement noise covariance. (dim_z, dim_z)
            f_args (tuple, optional): Additional arguments for the state transition function. Defaults to ().
            h_args (tuple, optional): Additional arguments for the measurement function. Defaults to ().

        Returns:
            loss (torch.Tensor): The loss value.

        Note:
            - h_args and f_args must scale with the time dimension.
            - Do not call this method for inference. Use batch_filtering or batch_smoothing instead.
        """
        # Process the measurements
        results = self.batch_filtering(
            z=z,
            x0=x0,
            P0=P0,
            Q=Q,
            R=R,
            f_args=f_args,
            h_args=h_args,
        )

        # The negative log-likelihood loss for grdient descent
        # The negative sign is used to convert the maximum likelihood problem to a minimization problem
        return -log_likelihood(
            results[self.TERMS["InnovationResidual"]],
            results[self.TERMS["InnovationCovariance"]],
        )


class VariationalNEKF(ParametricExtendedKalmanFilter):
    """The variational Neural Extended Kalman Filter (NEKF) model described in the paper."""

    def forward(
        self,
        z: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        Q: torch.Tensor | nn.Parameter,
        R: torch.Tensor | nn.Parameter,
        f_args: tuple = (),
        h_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """Processes the sequence of measurements to form an Variational Objective ELBO loss.

        Args:
            z (torch.Tensor): Measurement sequence. (T, dim_z)
            x0 (torch.Tensor): Initial state estimate. (dim_x, )
            P0 (torch.Tensor): Initial state covariance. (dim_x, dim_x)
            Q (torch.Tensor | nn.Parameter): Process noise covariance. (dim_x, dim_x)
            R (torch.Tensor | nn.Parameter): Measurement noise covariance. (dim_z, dim_z)
            f_args (tuple, optional): Additional arguments for the state transition function. Defaults to ().
            h_args (tuple, optional): Additional arguments for the measurement function. Defaults to ().

        Returns:
            loss (torch.Tensor): The loss value.

        Note:
            - h_args and f_args must scale with the time dimension.
            - Do not call this method for inference. Use batch_filtering or batch_smoothing instead.
        """
        # Process the measurements
        # Diable gradient computation of claculation of approximate posterior

        results = self.batch_smoothing(
            z=z,
            x0=x0,
            P0=P0,
            Q=Q,
            R=R,
            f_args=f_args,
            h_args=h_args,
        )

        # With gradient computation enabled, calculate the KL divergence and log likelihood
        kl_divergence = kl_divergence_of_transistion_model(
            Df=self._f,
            DF_args=f_args,
            x0=x0,
            P0=P0,
            x_smoothed=results[self.TERMS["SmoothedEstimate"]],
            P_smoothed=results[self.TERMS["SmoothedCovariance"]],
            Q=Q,
        )
        # Compute the log likelihood of the observation model
        log_likelihood = log_likelihood_of_observation_model(
            y=z,
            Dh=self._h,
            DH_args=h_args,
            x_smoothed=results[self.TERMS["SmoothedEstimate"]],
            P_smoothed=results[self.TERMS["SmoothedCovariance"]],
            R=R,
        )

        # Minimize the negative ELBO
        return kl_divergence - log_likelihood
