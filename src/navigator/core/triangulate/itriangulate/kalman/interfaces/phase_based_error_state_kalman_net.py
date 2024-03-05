"""GNSS Kalman Network Module.

This module implements a Kalman network for Global Navigation Satellite System (GNSS) applications,
specifically designed for precise positioning using error state Kalman network techniques.
It includes the ErrorStateKalmanNet class, which models the error state of a GNSS receiver
and estimates parameters such as position, velocity, clock drift, wet tropospheric delay, and biases.

Classes:
    - ErrorStateKalmanNet: Implements an error state Kalman network for GPS/GNSS.

State Definition:
    The state vector (curr_state) represents the receiver and includes:
    - Dx: Error in X coordinate from baseline coordinate.
    - x_dot: Velocity of the x-coordinate.
    - Dy: Error in Y coordinate from baseline coordinate.
    - y_dot: Velocity of the y-coordinate.
    - Dz: Error in Z coordinate from baseline coordinate.
    - z_dot: Velocity of the z-coordinate.
    - Dclock_drift: Error in clock drift from baseline coordinate.
    - clock_drift_rate: Clock drift rate.
    - wet_tropospheric_delay: Wet tropospheric delay.

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

from ......filters.kalman_nets.gru_knets.gru_extended_kalman_net import (
    GRUExtendedKalmanBlock,
)
from ......filters.kalman_nets.kalman_net_base import AbstractKalmanNet
from ...algos.combinations.range_combinations import SPEED_OF_LIGHT
from ..tools.phase_based_kalman_net.measurement import phase_measurement_model
from ..tools.phase_based_kalman_net.state_transistion import (
    phase_state_transistion_matrix,
)


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

    def __init__(
        self, dt: float, num_sv: int, base_line: torch.Tensor | None = None
    ) -> None:
        """Initializes the error state Kalman filter for GPS/GNSS.

        Args:
            dt (float): Time step of the filter.
            num_sv (int): Number of satellites to track.
            base_line (torch.Tensor, optional): Baseline coordinates for error estimation.
        """
        # Initialize the base class
        super().__init__(
            dim_state=9,  # Total 9 state elements + num_sv biases
            dim_measurement=2 * num_sv,
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
                1115077.69025948,
                -4843958.49112974,
                3983260.99261736,
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
        self.F = phase_state_transistion_matrix(dt=self.dt).to(
            device=self.device, dtype=self.dtype
        )

        # Initialize the hidden state
        self.reset()

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
        self.reset_internal_state(batch_dim=batch_size)

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
        return phase_measurement_model(
            error_state=x,
            sv_matrix=sv_matrix,
            base_line=self.base_line,
        )

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
