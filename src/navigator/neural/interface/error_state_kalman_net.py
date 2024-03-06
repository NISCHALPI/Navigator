"""GNSS Kalman Network Module.

This module implements a Kalman network for Global Navigation Satellite System (GNSS) applications,
specifically designed for precise positioning using error state Kalman network techniques.
It includes the ErrorStateKalmanNet class, which models the error state of a GNSS receiver
and estimates parameters such as position, velocity, clock drift, wet tropospheric delay, and biases.

Classes:
    - ErrorStateKalmanNet: Implements an error state Kalman network for GPS/GNSS.


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

Usage:
    >>> from gnss_kalman_network import ErrorStateKalmanNet

    >>> kalman_net = ErrorStateKalmanNet(dt=1.0, num_sv=4)
    >>> measurement = obtain_gnss_measurement()  # Replace with actual measurement function
    >>> kalman_net.predict()
    >>> kalman_net.update(measurement)

See Also:
    - Linear observation model for PPP:
      https://gssc.esa.int/navipedia/index.php?title=Linear_observation_model_for_PPP
"""

import torch
import torch.nn as nn

from ...core.triangulate.itriangulate.algos.combinations.range_combinations import (
    SPEED_OF_LIGHT,
)
from ..architectures.kalman_nets.gru_knets.gru_extended_kalman_net import (
    GRUExtendedKalmanBlock,
)
from ..architectures.kalman_nets.kalman_net_base import AbstractKalmanNet
from ..architectures.set_transformer import PMA, SAB
from .tools.measurement import kalman_net_measurement_model
from .tools.state_transistion import (
    kalman_net_state_transistion_model,
)

__all__ = ["ErrorStateKalmanNet"]

# TODO: Add the set transformer before feeding the data to the Kalman Net to improve the attention mechanism.


class ErrorStateKalmanNet(AbstractKalmanNet):
    """Implements the error state Kalman filter for GPS/GNSS.

    This class represents a Kalman filter designed for GPS/GNSS applications.
    It models the error state of the receiver and estimates various parameters such as
    position, velocity, clock drift, wet tropospheric delay, and biases.

    Attributes:
        dim_state (int): Dimension of the state vector.
        dim_measurement (int): Dimension of the measurement vector.
        dt (float): Time step of the filter.
        num_sv (int): Number of satellites to track.
        base_line (torch.Tensor): Baseline coordinates for iterative error estimation.
        gru_kalman_block (GRUExtendedKalmanBlock): GRU-based Kalman filter block.
        F (torch.Tensor): State transition matrix.


    See Also:
        Linear observation model for PPP:
        https://gssc.esa.int/navipedia/index.php?title=Linear_observation_model_for_PPP
    """

    PROJECTION_DIM = 30
    NUM_HEADS = 5
    SCALING = 1e7
    DIM_STATE = 7

    def __init__(
        self, dt: float, num_sv: int, base_line: torch.Tensor | None = None
    ) -> None:
        """Initializes the error state Kalman filter for GPS/GNSS.

        Args:
            dt (float): Time step of the filter.
            num_sv (int): Number of satellites to latently track.
            base_line (torch.Tensor, optional): Baseline coordinates for error estimation.
        """
        # Initialize the base class
        super().__init__(
            dim_state=self.DIM_STATE,  # Total 9 state elements + num_sv biases
            dim_measurement=num_sv,
            dt=dt,
            flavor=["F1", "F2", "F3", "F4"],
            max_history=5,
        )
        # Number of satellites to track
        if num_sv < 3:
            raise ValueError(
                "The number of satellites to track should be greater than 2."
            )

        self.num_sv = num_sv

        # Use coordinates of Washington DC as the initial guess for the state vector
        self.base_line = torch.tensor(
            [
                1115077.69025948,  # X_Coordinate
                10,  # X_Velocity 10 m/s
                -4843958.49112974,  #  Y_Coordinate,
                10,  # Y_Velocity 10 m/s
                3983260.99261736,  # Z_Coordinate
                5,  # Z_Velocity 5 m/s
                1e-5 * SPEED_OF_LIGHT,
            ],
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )

        if base_line is not None:
            # Check baseline is of the correct shape
            if base_line.shape == (4,):
                raise ValueError(
                    "The base line coordinates should be a 1D tensor := [x, y, z, clock_bias]."
                )
            self.base_line = base_line

        # Initialize the GRU gain network
        # Initialize a GRU cell
        self.network = GRUExtendedKalmanBlock(
            dim_state=self.dim_state,
            dim_measurement=self.dim_measurement,
            flavours=self.flavor,
        )

        # Initialize the state transition matrix
        self.F = kalman_net_state_transistion_model(dt=self.dt).to(
            device=self.device, dtype=self.dtype
        )

        # The Set Transformer attention mechanism for measurement
        # encoding before feeding to the Kalman Net
        # Initialize the encoder
        self.encoder = nn.Sequential(
            nn.Linear(5, self.PROJECTION_DIM),  # Initial Projection Layer
            SAB(
                dim_Q=self.PROJECTION_DIM,
                num_heads=self.NUM_HEADS,
                ln=True,
                fnn_dim=64,
            ),
            SAB(
                dim_Q=self.PROJECTION_DIM, num_heads=self.NUM_HEADS, ln=True, fnn_dim=64
            ),
        )
        self.decoder = nn.Sequential(
            PMA(
                dim_Q=self.PROJECTION_DIM,
                dim_S=self.num_sv,  # Latently track the number of satellites
                num_heads=self.NUM_HEADS,
                ln=True,
                fnn_dim=64,
            ),
            SAB(
                dim_Q=self.PROJECTION_DIM, num_heads=self.NUM_HEADS, ln=True, fnn_dim=64
            ),
        )
        self.rFF = nn.Sequential(
            nn.Linear(self.PROJECTION_DIM, self.PROJECTION_DIM),
            nn.ReLU(),
            nn.Linear(self.PROJECTION_DIM, self.PROJECTION_DIM // 2),
            nn.ReLU(),
            nn.Linear(self.PROJECTION_DIM // 2, 4),
        )
        # Initialize the hidden state
        self.reset()

        # Set learning rate
        self.learning_rate = 1e-3

    # Override the to function to copy the internal state of the filter to the device
    def to(self, *args, **kwargs) -> "ErrorStateKalmanNet":
        """Copies the internal state of the filter to the device."""
        # Call the parent to method
        super().to(*args, **kwargs)

        # Set the internal state
        self.Q_GRU_HIDDEN_STATE = self.Q_GRU_HIDDEN_STATE.to(*args, **kwargs)
        self.P_GRU_HIDDEN_STATE = self.P_GRU_HIDDEN_STATE.to(*args, **kwargs)
        self.S_GRU_HIDDEN_STATE = self.S_GRU_HIDDEN_STATE.to(*args, **kwargs)
        self.F = self.F.to(*args, **kwargs)
        self.base_line = self.base_line.to(*args, **kwargs)
        # Return self
        return self

    def reset(self, batch_size: int = 1) -> None:
        """Resets the internal state for kalman filtering."""
        self.Q_GRU_HIDDEN_STATE = torch.zeros(
            batch_size, self.network.Q_dim, dtype=self.dtype, device=self.device
        )
        self.P_GRU_HIDDEN_STATE = torch.zeros(
            batch_size, self.network.P_dim, dtype=self.dtype, device=self.device
        )
        self.S_GRU_HIDDEN_STATE = torch.zeros(
            batch_size, self.network.S_dim, dtype=self.dtype, device=self.device
        )

        # Reset the kalman filter state of parent
        self.reset(batch_dim=batch_size)

        return

    def calculate_kalman_gain(
        self, combinations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculates the Kalman gain for the given combinations of states and measurements.

        Args:
            combinations: A dictionary containing the combinations of states and measurements.

        Returns:
            The Kalman gain matrix
        """
        # Calculate the Kalman gain
        # Add the HIDEN_STATE To the combinations
        combinations[GRUExtendedKalmanBlock.Q_HIDDEN_KEY] = self.Q_GRU_HIDDEN_STATE
        combinations[GRUExtendedKalmanBlock.P_HIDDEN_KEY] = self.P_GRU_HIDDEN_STATE
        combinations[GRUExtendedKalmanBlock.S_HIDDEN_KEY] = self.S_GRU_HIDDEN_STATE

        # Calculate the Kalman gain
        output = self.network(combinations)

        # Update the hidden state
        self.Q_GRU_HIDDEN_STATE = output[GRUExtendedKalmanBlock.Q_HIDDEN_KEY]
        self.P_GRU_HIDDEN_STATE = output[GRUExtendedKalmanBlock.P_HIDDEN_KEY]
        self.S_GRU_HIDDEN_STATE = output[GRUExtendedKalmanBlock.S_HIDDEN_KEY]

        # Return the Kalman gain
        return output["KG"]

    def fx(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
        """The state transition function for the Phase-based error state kalman net.

        Args:
            x: The current state of the system.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The predicted state of the system forwad in time.
        """
        # The state transition function is a linear function of the state
        return torch.matmul(self.F, x)

    def hx(self, x: torch.Tensor, sv_matrix: torch.Tensor) -> torch.Tensor:
        """The measurement function for the Phase-based error state kalman net.

        Args:
            x: The current state of the system.
            sv_matrix: The satellite vehicle matrix.


        Returns:
            torch.Tensor: The predicted measurement of the system.
        """
        return kalman_net_measurement_model(
            state=x,
            sv_matrix=sv_matrix,
            base_line=self.base_line,
        )

    def _trajectory_set_transformer(
        self, x_trajectory: torch.Tensor, sv_trajectory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The set transformer for the error state kalman net.

        Args:
            x_trajectory: The trajectory of the measurements vector. (T, M)
            sv_trajectory: The trajectory of the satellites for each range measurement. (T, num_sv, 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The latent space code and measurements.

        Notation:
            T: Time steps trajectory
            M: Measurement dimension
            S: State dimension
        """
        # Stack the code and phase measurements in a single tensor (B, T, SV, 2) where sv = M/2
        x_trajectory = x_trajectory.view(
            x_trajectory.shape[0], x_trajectory.shape[1] // 2, 2
        )
        # Add the stack to the sv_trajectory tensoe
        # i.e the sv_trajectory tensor is now (B, T, SV, 2 + 3)
        sv_trajectory = torch.cat([x_trajectory, sv_trajectory], dim=-1)
        # Scale the measurements
        sv_trajectory = sv_trajectory / self.SCALING

        # Encode the measurements to a latent space
        latent = self.rFF(
            self.decoder(self.encoder(sv_trajectory))
        )  # See Set Transformer Paper

        # Interprete the latent space as the measurements
        # (B, T, num_sv, 4) where 4 crossponds [code, x, y, z]
        latent_code = latent[..., 0]
        latent_sv_matrix = latent[..., 1:]

        return latent_code, latent_sv_matrix

    def batch_encode(
        self, x_trajectory: torch.Tensor, sv_trajectory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch encode the trajectory of the measurements and satellite vehicles.

        Args:
            x_trajectory: The trajectory of the measurements vector. (B, T, M)
            sv_trajectory: The trajectory of the satellites for each range measurement. (B, T, num_sv, 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The latent space code [B, T, 1], and measurements [B, T, num_sv, 3]

        Notation:
            B: Batch size
            T: Time steps trajectory
            M: Measurement dimension
            S: State dimension
        """
        # Apply the set transformer to each trajectory and retain
        # the shape of the batch
        latent_code = []
        latent_sv_matrix = []
        for i in range(x_trajectory.shape[0]):
            code, sv_matrix = self._trajectory_set_transformer(
                x_trajectory[i], sv_trajectory[i]
            )
            latent_code.append(code)
            latent_sv_matrix.append(sv_matrix)

        return torch.stack(latent_code, dim=0), torch.stack(latent_sv_matrix, dim=0)

    def forward(
        self, x_trajectory: torch.Tensor, sv_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """The forward pass for the error state kalman net.

        Args:
            x_trajectory: The trajectory of the measurements vector. (B, T, M)
            sv_trajectory: The trajectory of the satellites for each range measurement. (B, T, num_sv, 3)

        Returns:
            torch.Tensor: The predicted state of the system by the kalman net. (B, T, S)

        Notation:
            B: Batch size
            T: Time steps
            M: Measurement dimension
            S: State dimension
        """
        # Reset the internal state
        self.reset(batch_size=x_trajectory.shape[0])

        # Run the kalman filter over the trajectory
        state = []

        # Batch Encode each trajectory to a latent space
        x_trajectory, sv_trajectory = self.batch_encode(
            x_trajectory=x_trajectory, sv_trajectory=sv_trajectory
        )

        # Run the kalman filter over the trajectory
        for i in range(x_trajectory.shape[1]):
            # Predict the state
            self.predict()

            # Update the state
            state.append(
                self.update(
                    y=x_trajectory[:, i, :],
                    hx_kwargs={"sv_matrix": sv_trajectory[:, i, :]},
                )
            )

        stacked_state = torch.stack(state, dim=1)

        # Reset the state just to be in single batch state
        self.reset()

        return stacked_state

    def training_step(
        self,
        batch: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        """The training step for the error state kalman net.

        Args:
            batch: The batch of training data.
            batch_idx: The index of the batch.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The loss of the training step.
        """
        true_state, x_trajectory, sv_trajectory = batch

        # Unsqueeze the data
        true_state, x_trajectory, sv_trajectory = (
            true_state.unsqueeze(0),
            x_trajectory.unsqueeze(0),
            sv_trajectory.unsqueeze(0),
        )

        # Forward pass
        state = self.forward(x_trajectory, sv_trajectory)

        # Compute the loss between the true state and the predicted state
        loss = torch.nn.functional.mse_loss(state, true_state)

        # Log the loss
        self.log("train_loss", loss)

        return {"loss": loss}

    def validation_step(
        self,
        batch: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        """The validation step for the error state kalman net.

        Args:
            batch: The batch of validation data.
            batch_idx: The index of the batch.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The loss of the validation step.
        """
        # Unpack the batch
        true_state, x_trajectory, sv_trajectory = batch
        # Unsqueeze the data
        true_state, x_trajectory, sv_trajectory = (
            true_state.unsqueeze(0),
            x_trajectory.unsqueeze(0),
            sv_trajectory.unsqueeze(0),
        )

        # Forward pass
        state = self.forward(x_trajectory, sv_trajectory)

        # Compute the loss between the true state and the predicted state
        loss = torch.nn.functional.mse_loss(state, true_state)

        # Log the loss
        self.log("val_loss", loss)

        return {"val_loss": loss}

    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """Configures the optimizer for the error state kalman net.

        Returns:
            dict[str, torch.optim.Optimizer]: The optimizer for the error state kalman net.
        """
        # Add the optimizer
        params = {
            "optimizer": torch.optim.Adam(self.parameters(), lr=self.learning_rate),
        }

        # Add the cosine annealing scheduler
        params["lr_scheduler"] = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                params["optimizer"],
                T_max=30,
                eta_min=1e-6,
            ),
            "interval": "step",
        }

        return params
