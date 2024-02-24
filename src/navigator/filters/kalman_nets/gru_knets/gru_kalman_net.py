"""Implemets the original KalmanNet using GRU cells.

This is the original implementation of the KalmanNet using GRU cells. The GRU cell tracks 
the joint state and measurement covariance. The Kalman gain is then computed using the
joint covariance.

Source:
    - G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun, and Y. C. Eldar,
      "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics,"
      in IEEE Transactions on Signal Processing, vol. 70, pp. 1532-1547, 2022, doi: 10.1109/TSP.2022.3158588.

"""

import torch
import torch.nn as nn

from .linear_blocks import LinearBlocks

__all__ = ["GRUKalmanBlock"]


class GRUKalmanBlock(nn.Module):
    """The original KalmanNet GRU block.

    This class implements the original KalmanNet, a variant of the Kalman filter, using GRU cells. It tracks the
    joint state and measurement covariance, and computes the Kalman gain using the joint covariance.

    Args:
        dim_state (int): The dimension of the state space.
        dim_measurement (int): The dimension of the measurement space.
        flavours (list[str]): The list of flavours for the KalmanNet. See the paper for more details.
        integer_scaler (int): The integer scaler for the hidden dimension of the GRU cell. Default is 10.
    """

    def __init__(
        self,
        dim_state: int,
        dim_measurement: int,
        flavours: list[str],
        integer_scaler: int = 10,
    ) -> None:
        """The base KalmanNet class for Kalman Filtering.

        Args:
            dim_state (int): The dimension of the state space.
            dim_measurement (int): The dimension of the measurement space.
            flavours (list[str]): The list of flavours for the KalmanNet. See the paper for more details.
            integer_scaler (int): The integer scaler for the hidden dimension of the GRU cell. Default is 10.
        """
        super().__init__()

        # Check the validity of the input dimension
        if dim_state <= 0:
            raise ValueError(f"Invalid dim_state: {dim_state}")
        if dim_measurement <= 0:
            raise ValueError(f"Invalid dim_measurement: {dim_measurement}")
        if integer_scaler <= 0:
            raise ValueError(f"Invalid integer_scaler: {integer_scaler}")

        # Store the input parameters
        self._dim_state = dim_state
        self._dim_measurement = dim_measurement
        self._integer_scaler = integer_scaler
        self._flavours = flavours

        # Flavours dictionary
        self.flavour_dict = {
            "F1": dim_measurement,
            "F2": dim_measurement,
            "F3": dim_state,
            "F4": dim_state,
        }
        # Initialize the input layer that maps the flavours to the input dimension
        self.input_layer = LinearBlocks(
            input_dim=sum([self.flavour_dict[flavour] for flavour in self._flavours]),
            output_dim=(dim_state**2 + dim_measurement**2),
            hidden_dim=(dim_state**2 + dim_measurement**2),
            layers=3,
        )
        # Initialize the GRU cell for joint tracking of measurement covariance and state covariance
        self.network = nn.GRUCell(
            input_size=(dim_state**2 + dim_measurement**2),
            hidden_size=integer_scaler * (dim_state**2 + dim_measurement**2),
        )
        # Initialize the output layer that maps the GRU cell output to Kalman gain
        self.output_layer = LinearBlocks(
            input_dim=integer_scaler * (dim_state**2 + dim_measurement**2),
            output_dim=dim_state
            * dim_measurement,  # Kalman gain dimension is (dim_state x dim_measurement)
            hidden_dim=(dim_state**2 + dim_measurement**2),
            layers=3,
        )

    def forward(
        self, combinations: dict[str, torch.Tensor], hx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the KalmanNet.

        Args:
            combinations (dict[str, torch.Tensor]): The dictionary of combinations of the input tensors. [F1, F2, F3, F4]
            hx (torch.Tensor): The hidden state tensor for the GRU cell.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The kalman gain tensor and the hidden state tensor for next time step.
        """
        # Order the dictionary by keys
        # Essentially, this is a sanity check to ensure that the input dictionary is ordered in sucessive runs
        combinations = dict(sorted(combinations.items(), key=lambda item: item[0]))

        # Hstack the input tensors to form the input tensor
        x = torch.hstack([combinations[flavour] for flavour in self._flavours])

        # Pass through the input layer
        x = self.input_layer(x)
        # Pass through the GRU cell
        hx = self.network(x, hx)

        # Pass through the output layer
        x = self.output_layer(hx)

        return x.view(-1, self._dim_state, self._dim_measurement), hx

    @property
    def hx_dim(self) -> int:
        """The dimension of the measurement space.

        Returns:
            int: The dimension of the measurement space.
        """
        return self._integer_scaler * (
            self._dim_state**2 + self._dim_measurement**2
        )

    @hx_dim.setter
    def hx_dim(self, value: int):  # noqa : ARG003
        """The setter for the dimension of the measurement space.

        Args:
            value (int): The new dimension of the measurement space.
        """
        raise AttributeError(
            "Use the constructor to set the dimension of the measurement space."
        )


# Path: src/navigator/filters/kalman_nets/gru_knets/gru_kalman_net.py
