"""Implementation of a parametrized filter model for GNSS triangulation and state estimation.

This module provides a PyTorch Lightning module `TriangulationFilter` that utilizes 
a Parametric Extended Kalman Filter (PEKF) for processing GNSS (Global Navigation 
Satellite System) data. It includes methods for prediction, update, batch filtering, 
and batch smoothing of the state vector and covariance matrix.

The filter model supports customization of:
    - Transition model (`f`) for state propagation (default: constant velocity model)
    - Observation model (`h`) for measurement prediction (default: GNSS observation model)
    - Process noise covariance matrix (`Q`) for modeling uncertainty in state transitions (default: SymmetricPositiveDefiniteMatrix(I))
    - Observation noise covariance matrix (`R`) for modeling uncertainty in measurements (default: SymmetricPositiveDefiniteMatrix(I))

Attributes:
    DIM_X (int): Dimensionality of the state vector.
    STATE_NAMES (list[str]): Names of state vector components.

Methods:
    predict: Predicts the state vector and covariance matrix given current state.
    update: Updates the state vector and covariance matrix using current observation.
    batch_filtering: Performs batch filtering over a sequence of observations.
    batch_smoothing: Implements fixed-interval smoothing over a sequence of observations.
    forward: Computes the negative log likelihood of the model given observations.

Usage:
    >>> model = TriangulationFilter(dt=0.1, dim_z=3)
    >>> x0 = torch.zeros(model.DIM_X)
    >>> P0 = torch.eye(model.DIM_X)
    >>> z = torch.randn(10, model.dim_z)
    >>> sv_pos = torch.randn(10, model.dim_z, 3)
    >>> smoothed_state = model(z, x0, P0, sv_pos)
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..architectures.parametric_ekf.ekf import (
    NegativeLogLikelihoodPEKF,
    ParametricExtendedKalmanFilter,
    VariationalNEKF,
)
from .dynamics_model.constant_velocity_triangulation_model import (
    ObservationModel,
    SymmetricPositiveDefiniteMatrix,
    TransitionModel,
)

__all__ = ["ParametricExtendedInterface"]


class ParametricExtendedInterface(pl.LightningModule):
    """A PyTorch module implementing a parametrized filter model for GNSS triangulation and state estimation.

    This class utilizes a Parametric Extended Kalman Filter (PEKF) to process GNSS data. It includes methods for
    predicting, updating, batch filtering, and batch smoothing of the state vector and covariance matrix.

    The filter model supports customization of:
        - Transition model (`f`) for state propagation (default: constant velocity model)
        - Observation model (`h`) for measurement prediction (default: GNSS observation model)
        - Process noise covariance matrix (`Q`) for modeling uncertainty in state transitions (default: SymmetricPositiveDefiniteMatrix(I))
        - Observation noise covariance matrix (`R`) for modeling uncertainty in measurements (default: SymmetricPositiveDefiniteMatrix(I))

    Attributes:
        DIM_X (int): Dimensionality of the state vector.
        STATE_NAMES (list[str]): Names of state vector components.
        lr_keys (list[str]): Keys for learning rate adjustment.
        NEGATIVE_LOG_LIKELIHOOD (str): Constant for negative log likelihood objective.
        VARIATIONAL_INFERENCE (str): Constant for variational inference objective.

    Methods:
        __init__(dt, dim_z, f, h, Q, R, objective, lr):
            Initializes the filter model.

        set_objective(objective):
            Sets the objective function for training.

        predict(x, P):
            Predicts the state vector and covariance matrix given current state.

        update(z, x_prior, P_prior, sv_pos):
            Updates the state vector and covariance matrix using current observation.

        batch_filtering(z, x0, P0, sv_pos):
            Performs batch filtering over a sequence of observations.

        batch_smoothing(z, x0, P0, sv_pos):
            Implements fixed-interval smoothing over a sequence of observations.

        forward(z, x0, P0, sv_pos):
            Computes the negative log likelihood of the model given observations.

        parametric_filter:
            Property for getting/setting the underlying filter model.

        Q:
            Property for getting/setting the process noise covariance matrix.

        R:
            Property for getting/setting the observation noise covariance matrix.

        tune(z, x0, P0, sv_pos, lr, max_epochs, clip_grad_norm, log_interval, objective):
            Tunes the model parameters using training data.
    """

    DIM_X = 8
    STATE_NAMES = [
        "x",
        "x_dot",
        "y",
        "y_dot",
        "z",
        "z_dot",
        "cdt",
        "cdt_dot",
    ]

    lr_keys = ["f", "h", "Q", "R"]

    NEGATIVE_LOG_LIKELIHOOD = "negative_log_likelihood"
    VARIATIONAL_INFERENCE = "variational_inference"

    def __init__(
        self,
        dt: float = 1.0,
        dim_z: int = 5,
        f: nn.Module | None = None,
        h: nn.Module | None = None,
        Q: nn.Module | None = None,
        R: nn.Module | None = None,
        objective: str = "negative_log_likelihood",
    ) -> None:
        """Initialize the parametrized filter model.

        Args:
            dt (float, optional): The time step. Defaults to 1.0.
            dim_z (int, optional): The dimension of the observation vector. Defaults to 5.
            f (nn.Module, optional): The transition model. Defaults to None.
            h (nn.Module, optional): The observation model. Defaults to ObservationModel().
            Q (nn.Module, optional): The process noise covariance matrix. Defaults to None.
            R (nn.Module, optional): The observation noise covariance matrix. Defaults to None.
            objective (str, optional): The objective function. Defaults to "negative_log_likelihood".
        """
        super().__init__()

        # Initialize the process noise covariance matrix
        self.Q_module = (
            SymmetricPositiveDefiniteMatrix(
                M=torch.eye(self.DIM_X),
            )
            if Q is None
            else Q
        )

        self.R_module = (
            SymmetricPositiveDefiniteMatrix(
                M=torch.eye(dim_z),
            )
            if R is None
            else R
        )

        # Store the transition and observation models
        self.f = f if f is not None else TransitionModel(dt=dt, learnable=False)
        self.h = h if h is not None else ObservationModel()

        # Store the dimension of the observation vector
        self.dim_z = dim_z

        # Set the objective function
        self.set_objective(objective)

    def set_objective(self, objective: str) -> None:
        """Set the objective function for training the model.

        Args:
            objective (str): The objective function.
        """
        if objective == self.NEGATIVE_LOG_LIKELIHOOD:
            self.parametric_filter = NegativeLogLikelihoodPEKF(
                dim_x=self.DIM_X,
                dim_z=self.dim_z,
                f=self.f,
                h=self.h,
            )
        elif objective == self.VARIATIONAL_INFERENCE:
            self.parametric_filter = VariationalNEKF(
                dim_x=self.DIM_X,
                dim_z=self.dim_z,
                f=self.f,
                h=self.h,
            )
        else:
            raise ValueError(
                f"Objective function {objective} not supported. Choose from {self.NEGATIVE_LOG_LIKELIHOOD}, {self.VARIATIONAL_INFERENCE}"
            )

    def _data_to(self, *args) -> tuple[torch.Tensor]:
        """Take the data to the device and dtype.

        Args:
            args: The data tensors.

        Returns:
            tuple[torch.Tensor]: The data tensors on the device and dtype.
        """
        return tuple(arg.to(device=self.device, dtype=self.dtype) for arg in args)

    def predict(
        self,
        x: torch.Tensor,
        P: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the state vector.

        Args:
            x (torch.Tensor): The state vector. (dim_x,)
            P (torch.Tensor): The covariance matrix. (dim_x, dim_x)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The predicted state vector and covariance matrix.
        """
        pred_outs = self.parametric_filter.predict(
            x_posterior=x, P_posterior=P, Q=self.Q, f_args=()
        )

        return (
            pred_outs[ParametricExtendedKalmanFilter.TERMS["PredictedEstimate"]],
            pred_outs[ParametricExtendedKalmanFilter.TERMS["PredictedCovariance"]],
        )

    def update(
        self,
        z: torch.Tensor,
        x_prior: torch.Tensor,
        P_prior: torch.Tensor,
        sv_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update the state vector.

        Args:
            z (torch.Tensor): The observation vector. (dim_z,)
            x_prior (torch.Tensor): The prior state vector. (dim_x,)
            P_prior (torch.Tensor): The prior covariance matrix. (dim_x, dim_x)
            sv_pos (torch.Tensor): The satellite positions. (dim_z,3)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The updated state vector, covariance matrix, and innovation.
        """
        update_outs = self.parametric_filter.update(
            z=z, x_prior=x_prior, P_prior=P_prior, R=self.R, h_args=(sv_pos,)
        )

        return (
            update_outs[ParametricExtendedKalmanFilter.TERMS["UpdatedEstimate"]],
            update_outs[ParametricExtendedKalmanFilter.TERMS["UpdatedCovariance"]],
            update_outs[ParametricExtendedKalmanFilter.TERMS["Innovation"]],
        )

    def batch_filtering(
        self,
        z: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        sv_pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Processes the GNSS data using the EKF.

        Args:
            z (torch.Tensor): The observation vector. (seq_len, dim_z)
            x0 (torch.Tensor): The initial state vector. (dim_x,)
            P0 (torch.Tensor): The initial covariance matrix. (dim_x, dim_x)
            sv_pos (torch.Tensor): The satellite positions. (seq_len, dim_z, 3)

        Returns:
            dict[str, torch.Tensor]: The smoothed state vector ,covariance matrix, other intermediate variables.
        """
        # Take the data to the device and dtype
        z, x0, P0, sv_pos = self._data_to(z, x0, P0, sv_pos)
        return self.parametric_filter.batch_filtering(
            z=z,
            x0=x0,
            P0=P0,
            Q=self.Q,
            R=self.R,
            f_args=(),
            h_args=(sv_pos,),
        )

    def batch_smoothing(
        self,
        z: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        sv_pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Implments fixed-interval smoothing using the EKF for time-series GNSS data.

        Args:
            z (torch.Tensor): The observation vector. (seq_len, dim_z)
            x0 (torch.Tensor): The initial state vector. (dim_x,)
            P0 (torch.Tensor): The initial covariance matrix. (dim_x, dim_x)
            sv_pos (torch.Tensor): The satellite positions. (seq_len, dim_z, 3)

        Returns:
            dict[str, torch.Tensor]: The smoothed state vector ,covariance matrix, other intermediate variables.
        """
        # Take the data to the device and dtype
        z, x0, P0, sv_pos = self._data_to(z, x0, P0, sv_pos)
        if self.training:
            return self.parametric_filter.batch_smoothing(
                z=z,
                x0=x0,
                P0=P0,
                Q=self.Q,
                R=self.R,
                f_args=(),
                h_args=(sv_pos,),
            )

        with torch.no_grad():
            return self.parametric_filter.batch_smoothing(
                z=z,
                x0=x0,
                P0=P0,
                Q=self.Q,
                R=self.R,
                f_args=(),
                h_args=(sv_pos,),
            )

    def forward(
        self,
        z: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        sv_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the negative log likelihood of the model at current state.

        Args:
            z (torch.Tensor): The observation vector. (seq_len, dim_z)
            x0 (torch.Tensor): The initial state vector. (dim_x,)
            P0 (torch.Tensor): The initial covariance matrix. (dim_x, dim_x)
            sv_pos (torch.Tensor): The satellite positions. (seq_len, dim_z, 3)

        Returns:
            torch.Tensor: The smoothed state vector.
        """
        # Take the data to the device and dtype
        z, x0, P0, sv_pos = self._data_to(z, x0, P0, sv_pos)
        return self.parametric_filter(
            z=z,
            x0=x0,
            P0=P0,
            Q=self.Q,
            R=self.R,
            f_args=(),
            h_args=(sv_pos,),
        )

    @property
    def parametric_filter(self) -> ParametricExtendedKalmanFilter:
        """The underlying filter model.

        Returns:
            ParametricExtendedKalmanFilter: The underlying filter model.
        """
        return self._filter

    @parametric_filter.setter
    def parametric_filter(self, value: ParametricExtendedKalmanFilter) -> None:
        """Set the underlying filter model.

        Args:
            value (ParametricExtendedKalmanFilter): The new filter model.
        """
        # If the filter model is changed, update the objective function
        if not isinstance(value, ParametricExtendedKalmanFilter):
            raise ValueError(
                "The filter model must be an instance of ParametricExtendedKalmanFilter."
            )

        self._filter = value

    @property
    def Q(self) -> torch.Tensor:
        """The process noise covariance matrix.

        Returns:
            torch.Tensor: The process noise covariance matrix.
        """
        return self.Q_module()

    @property
    def R(self) -> torch.Tensor:
        """The observation noise covariance matrix.

        Returns:
            torch.Tensor: The observation noise covariance matrix.
        """
        return self.R_module()

    @Q.setter
    def Q(self, value: torch.Tensor) -> None:
        """Set the process noise covariance matrix.

        Args:
            value (torch.Tensor): The new process noise covariance matrix.
        """
        self.Q_module = SymmetricPositiveDefiniteMatrix(M=value)

    @R.setter
    def R(self, value: torch.Tensor) -> None:
        """Set the observation noise covariance matrix.

        Args:
            value (torch.Tensor): The new observation noise covariance matrix.
        """
        self.R_module = SymmetricPositiveDefiniteMatrix(M=value)

    def _grad_norms(self, module: nn.Module) -> list[float]:
        """Get the gradient norms for the parameters of a module.

        Args:
            module (nn.Module): The module.

        Returns:
            list[float]: The gradient norms.
        """
        grad_parms = [
            param.grad.detach().flatten()
            for param in module.parameters()
            if param.grad is not None
        ]

        if len(grad_parms) > 0:
            return torch.cat(grad_parms).norm()
        return 0.0

    def tune(
        self,
        z: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        sv_pos: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        monitor: bool = False,
        adjust_after: int = 0,
        max_epochs: int = 100,
        clip_grad_norm: float = 10000.0,
        log_interval: int = 1,
        objective: str = "negative_log_likelihood",
    ) -> None:
        """Tune the model parameters using the training data.

        This method trains the model by adjusting its parameters based on the provided training data.
        It uses the specified objective function to compute the loss and performs gradient descent optimization
        to update the model parameters.

        Args:
            z (torch.Tensor): The observation vector. (seq_len, dim_z)
            x0 (torch.Tensor): The initial state vector. (dim_x,)
            P0 (torch.Tensor): The initial covariance matrix. (dim_x, dim_x)
            sv_pos (torch.Tensor): The satellite positions. (seq_len, dim_z, 3)
            optimizer (torch.optim.Optimizer, optional): The optimizer to use for training. Defaults to None.
            monitor (bool, optional): Whether to monitor loss for learning rate adjustment. Defaults to False.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler. Defaults to None.
            adjust_after (int, optional): The number of epochs after which to adjust the learning rate. Defaults to 0.
            max_epochs (int, optional): The maximum number of epochs. Defaults to 100.
            clip_grad_norm (float, optional): The maximum gradient norm for gradient clipping. Defaults to 100.0.
            log_interval (int, optional): The interval at which to log the loss and gradient norm. Defaults to 1.
            objective (str, optional): The objective function to use for computing the loss. Defaults to "negative_log_likelihood".

        Returns:
            None

        """
        # Set the objective function
        self.set_objective(objective)

        # Set the losses
        losses = []
        grad_norms = []
        # Get the optimizer
        optimizer = (
            torch.optim.SGD(
                self.parameters(),
                lr=0.001,
                momentum=0.9,
            )
            if optimizer is None
            else optimizer
        )

        # Take the data to the device and dtype
        z, x0, P0, sv_pos = self._data_to(z, x0, P0, sv_pos)

        # Train the model
        for epoch in range(max_epochs):
            # Zero the gradients
            optimizer.zero_grad()

            # Compute the loss
            loss = self(
                z=z,
                x0=x0,
                P0=P0,
                sv_pos=sv_pos,
            )

            losses.append(loss.item())  # Store the loss
            loss.backward()  # Backpropagate the loss

            # Calculate the gradient norm
            Q_grad_norm = self._grad_norms(self.Q_module)
            R_grad_norm = self._grad_norms(self.R_module)
            f_grad_norm = self._grad_norms(self.f)
            h_grad_norm = self._grad_norms(self.h)

            # Print the loss and gradient norm
            if epoch % log_interval == 0:
                print(
                    f"Epoch {epoch}: Loss = {loss.item()}, Q_grad_norm = {Q_grad_norm}, R_grad_norm = {R_grad_norm}, f_grad_norm = {f_grad_norm}, h_grad_norm = {h_grad_norm}"
                )
            grad_norms.append(sum([Q_grad_norm, R_grad_norm, f_grad_norm, h_grad_norm]))

            # Clip the gradient norm
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)

            # Update the parameters
            optimizer.step()

            # If a learning rate scheduler is provided, update the learning rate
            if lr_scheduler is not None and epoch >= adjust_after:
                if monitor:
                    lr_scheduler.step(loss)
                else:
                    lr_scheduler.step()

        return {
            "losses": losses,
            "grad_norms": grad_norms,
        }
