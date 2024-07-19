"""Implements a constant velocity dynamics model for Kalman filter used in triangulation.

In the context of GNSS (Global Navigation Satellite System) triangulation, the dynamics model 
predicts the future state estimates of the target. The state vector is defined as:

x = [x, x_dot, y, y_dot, z, z_dot, cdt, cdt_dot]

where:
- x, y, z are the position coordinates of the target in the ECEF (Earth-Centered, Earth-Fixed) frame.
- x_dot, y_dot, z_dot are the velocities along the respective axes in the ECEF frame.
- cdt is the clock drift of the target.
- cdt_dot is the rate of change of clock drift.

Functions:
- G(dt: float) -> torch.Tensor:
  Returns the state transition matrix for the constant velocity model.

- h(x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
  Returns the measurement matrix for the constant velocity model.

- HJacobian(x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
  Returns the Jacobian matrix of the measurement matrix for the constant velocity model.

- Q(dt: float, Q_0: torch.Tensor) -> torch.Tensor:
  Returns the process noise matrix for the constant velocity model.

Classes:
- ObservationModel(nn.Module):
  A PyTorch module that implements the observation model for the constant velocity model.

- TransitionModel(nn.Module):
  A PyTorch module that implements the transition model for the constant velocity model.

- SymmetricPositiveDefiniteMatrix(nn.Module):
  A PyTorch module that ensures the input matrix remains symmetric and positive definite.

Usage:
>>> from constant_velocity_triangulation_model import G, h, HJacobian, Q, ObservationModel, TransitionModel, SymmetricPositiveDefiniteMatrix
>>> import torch

# Example usage of the functions
>>> dt = 0.1
>>> A = G(dt)
>>> x = torch.zeros(8)
>>> sv_pos = torch.randn(5, 3)
>>> measurement = h(x, sv_pos)
>>> jacobian = HJacobian(x, sv_pos)
>>> process_noise = Q(dt, torch.eye(8))

# Example usage of the classes
>>> observation_model = ObservationModel()
>>> transition_model = TransitionModel(dt=0.1)
>>> Q_matrix = SymmetricPositiveDefiniteMatrix(Q_0=torch.eye(8))
>>> Q_matrix_output = Q_matrix()

"""

import torch
import torch.nn as nn

__all__ = [
    "G",
    "h",
    "HJacobian",
    "discretized_process_noise_matrix",
    "ObservationModel",
    "TransitionModel",
    "SymmetricPositiveDefiniteMatrix",
    "DiagonalSymmetricPositiveDefiniteMatrix",
    "BiasedObservationModel",
]


def G(dt: float) -> torch.Tensor:
    """Returns the state transition matrix for the constant velocity model.

    Args:
        dt (float): Time step.

    Returns:
        torch.Tensor: State transition matrix.
    """
    return torch.Tensor(
        [
            [1, dt, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, dt, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )


def h(x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
    """Returns the measurement matrix for the constant velocity model.

    Args:
        x (torch.Tensor): State vector. (8,)
        sv_pos (torch.Tensor): Satellite position. (n, 3)

    Returns:
        torch.Tensor: Measurement matrix.
    """
    pos = x[[0, 2, 4]]
    return torch.linalg.norm(pos - sv_pos, dim=1) + x[6]


def HJacobian(x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
    """Returns the Jacobian of the measurement matrix for the constant velocity model.

    Args:
        x (torch.Tensor): State vector. (8,)
        sv_pos (torch.Tensor): Satellite position. (n, 3)

    Returns:
        torch.Tensor: Jacobian of the measurement matrix.
    """
    pos = x[[0, 2, 4]]
    diff = pos - sv_pos
    norm = torch.linalg.norm(diff, dim=1)

    # Initialize the Jacobian matrix
    HJ = torch.zeros((sv_pos.shape[0], 8))

    # Add the derivative of the measurement matrix with respect to position
    HJ[:, [0, 2, 4]] = diff / norm[:, None]

    # Add the derivative of the measurement matrix with respect to clock drift
    HJ[:, 6] = 1

    return HJ


def discretized_process_noise_matrix(dt: float, Q_0: torch.Tensor) -> torch.Tensor:
    """Returns the discretized process noise matrix for the constant velocity model.

    Args:
        dt (float): Time step.
        Q_0 (torch.Tensor): Process noise power spectral density matrix.

    Returns:
        torch.Tensor: Process noise matrix.
    """
    A = torch.tensor([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    F = torch.kron(torch.eye(len(Q_0) // 2), A).double()

    return F @ Q_0 @ F.T


class ObservationModel(nn.Module):
    """The observation model for the GNSS triangulation."""

    def __init__(self, trainable: bool = False, dim_measurement: int = 8) -> None:
        """Initializes the observation model.

        Args:
            trainable (bool): Flag to indicate whether the observation model is trainable. Default: False.
            dim_measurement (int): Dimension of the measurement. Default: 8.

        Note:
            The observation model can be parametrized by a neural network to learn the dynamics of the system.
        """
        super().__init__()

        if trainable:
            self.W = nn.Sequential(
                nn.Linear(8, dim_measurement),
                nn.ReLU(),
                nn.Linear(dim_measurement, dim_measurement),
                nn.ReLU(),
                nn.Linear(dim_measurement, dim_measurement),
            )

            # Initialzie the parameters to be zero
            for param in self.W.parameters():
                # Initialize the bias to zero
                if param.dim() == 1:
                    param.data.zero_()

                # Initialize the weights to kaming normal
                else:
                    nn.init.kaiming_normal_(param.data)

                # Make the weights very small to the transition behaves like a constant velocity model at the start of training
                param.data *= 1e-30

    def forward(self, x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
        """Forward pass of the observation model.

        Args:
            x (torch.Tensor): State vector. (8,)
            sv_pos (torch.Tensor): Satellite position. (n, 3)

        Returns:
            torch.Tensor: Predicted measurements.
        """
        return h(x, sv_pos) + self.W(x) if hasattr(self, "W") else h(x, sv_pos)


class BiasedObservationModel(nn.Module):
    """The observation model for the GNSS triangulation."""

    def __init__(self, trainable: bool = False, dim_measurement: int = 8) -> None:
        """Initializes the observation model.

        Args:
            trainable (bool): Flag to indicate whether the observation model is trainable. Default: False.
            dim_measurement (int): Dimension of the measurement. Default: 8.

        Returns:
            None
        """
        super().__init__()

        if trainable:
            self.W = nn.Parameter(torch.zeros(dim_measurement), requires_grad=True)

    def forward(self, x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
        """Forward pass of the observation model.

        Args:
            x (torch.Tensor): State vector. (8,)
            sv_pos (torch.Tensor): Satellite position. (n, 3)

        Returns:
            torch.Tensor: Predicted measurements.
        """
        return h(x, sv_pos) + self.W


class TransitionModel(nn.Module):
    """The transition model for the GNSS triangulation.

    Note:
        The transistion model can be parametrized by a neural network to learn the dynamics of the system.
    """

    def __init__(self, dt: float, learnable: bool = False) -> None:
        """Initializes the transition model.

        Args:
            dt (float): Time step.
            learnable (bool): Flag to indicate whether the transition model is trainable. Default: False.

        Returns:
            None
        """
        super().__init__()
        self.dt = dt
        self.register_buffer("F", G(dt))

        # Learnable Linear Layer
        if learnable:
            self.W = nn.Sequential(
                nn.Linear(8, 8),
                nn.GELU(),
                nn.Linear(8, 8),
                nn.GELU(),
                nn.Linear(8, 8),
            )

            # Initialzie the parameters to be zero
            for param in self.W.parameters():
                # Initialize the bias to zero
                if param.dim() == 1:
                    param.data.zero_()

                # Initialize the weights to kaming normal
                else:
                    nn.init.kaiming_normal_(param.data)

                # Make the weights very small to the transition behaves like a constant velocity model at the start of training
                param.data *= 1e-30

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transition model.

        Args:
            x (torch.Tensor): State vector. (8,)

        Returns:
            torch.Tensor: Predicted next state vector.
        """
        return self.F @ x + self.W(x) if hasattr(self, "W") else self.F @ x


class SymmetricPositiveDefiniteMatrix(nn.Module):
    """Module for ensuring a symmetric positive definite matrix using a parameterized approach.

    This module constructs a symmetric positive definite matrix from an initial matrix Q_0.
    It ensures that the resultant matrix is symmetric and has positive eigenvalues.

    Attributes:
        LT_mask (torch.Tensor): Lower triangular mask used for parameter initialization.
        W (nn.Parameter): Parameter representing the input matrix adjusted for symmetry and positivity.

    Methods:
        forward():
            Performs the forward pass of the module.
            Returns a symmetric positive definite matrix derived from the input parameter.
    """

    def __init__(self, M: torch.Tensor, trainable: bool = True) -> None:
        """Initializes the SymmetricPositiveDefiniteMatrix module.

        Args:
            M (torch.Tensor): Initial matrix for the parameter.
            trainable (bool): Flag to indicate whether the parameter is trainable. Default: True.
        """
        super().__init__()

        # Cholesky decomposition of the initial matrix
        L = torch.linalg.cholesky(M, upper=False)

        # Initialize the parameter with the lower triangular part of the Cholesky decomposition
        self.W = nn.Parameter(L, requires_grad=trainable)

        # Mask for the diagonal entries
        self.diag_mask = torch.eye(M.shape[0], dtype=torch.bool)

    def forward(self) -> torch.Tensor:
        """Forward pass of the SymmetricPositiveDefiniteMatrix module.

        Returns:
            torch.Tensor: Symmetric positive definite matrix derived from the input parameter.
        """
        # Make the diagonal entries positive
        L = torch.tril(self.W)
        L[self.diag_mask] = torch.abs(L[self.diag_mask])

        return L @ L.T


class DiagonalSymmetricPositiveDefiniteMatrix(nn.Module):
    """A PyTorch module representing a diagonal symmetric positive definite matrix.

    This module takes a diagonal matrix `M` as input and constructs a diagonal symmetric positive definite matrix `W`.
    The diagonal entries of `W` are initialized with the diagonal entries of `M` and can be trained if `trainable` is set to True.

    Args:
        M (torch.Tensor): The diagonal matrix used to initialize the diagonal entries of `W`.
        trainable (bool, optional): Whether the diagonal entries of `W` should be trainable. Defaults to True.

    Returns:
        torch.Tensor: The diagonal symmetric positive definite matrix `W`.
    """

    def __init__(self, M: torch.Tensor, trainable: bool = True) -> None:
        """Initializes the DiagonalSymmetricPositiveDefiniteMatrix module.

        Args:
            M (torch.Tensor): The diagonal matrix used to initialize the diagonal entries of `W`.
            trainable (bool): Flag to indicate whether the diagonal entries of `W` are trainable. Default: True.

        Returns:
            None
        """
        super().__init__()

        self.W = nn.Parameter(torch.diag(M.diagonal()), requires_grad=trainable)

    def forward(self) -> torch.Tensor:
        """Forward pass of the module.

        Returns:
            torch.Tensor: The diagonal symmetric positive definite matrix `W`.
        """
        # Make the diagonal entries positive
        return torch.abs(self.W)
