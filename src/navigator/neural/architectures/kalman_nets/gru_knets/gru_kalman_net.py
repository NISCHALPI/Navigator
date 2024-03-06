"""Implemets the original KalmanNet using GRU cells.

This is the original implementation of the KalmanNet using GRU cells. The GRU cell tracks 
the joint state and measurement covariance. The Kalman gain is then computed using the
joint covariance.

Source:
    - G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun, and Y. C. Eldar,
      "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics,"
      in IEEE Transactions on Signal Processing, vol. 70, pp. 1532-1547, 2022, doi: 10.1109/TSP.2022.3158588.


Convention:
    B - Batch size
    D - Dimension of the state space
    M - Dimension of the measurement space
"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from .linear_blocks import LinearBlocks

__all__ = ["GRUKalmanBlock"]


class GRUKalmanBlock(LightningModule):
    """The original KalmanNet GRU block.

    This class implements the original KalmanNet, a variant of the Kalman filter, using GRU cells. It tracks the
    joint state and measurement covariance, and computes the Kalman gain using the joint covariance.

    Args:
        dim_state (int): The dimension of the state space.
        dim_measurement (int): The dimension of the measurement space.
        layers (int): The number of GRU layers. Default is 1.
        flavours (list[str]): The list of flavours for the KalmanNet. See the paper for more details.
        integer_scaler (int): The integer scaler for the hidden dimension of the GRU cell. Default is 10.
    """

    HIDDEN_KEY = "hx"

    def __init__(
        self,
        dim_state: int,
        dim_measurement: int,
        flavour: list[str],
        gru_layers: int = 1,
        integer_scaler: int = 10,
    ) -> None:
        """The base KalmanNet class for Kalman Filtering.

        Args:
            dim_state (int): The dimension of the state space.
            dim_measurement (int): The dimension of the measurement space.
            flavour (list[str]): The list of flavours for the KalmanNet. See the paper for more details.
            gru_layers (int): The number of GRU layers. Default is 1.
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
        self._flavours = flavour
        self._gru_layers = gru_layers

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
        self.network = nn.GRU(
            input_size=(dim_state**2 + dim_measurement**2),
            hidden_size=integer_scaler * (dim_state**2 + dim_measurement**2),
            num_layers=gru_layers,
        )
        # Initialize the output layer that maps the GRU cell output to Kalman gain
        self.output_layer = LinearBlocks(
            input_dim=integer_scaler * (dim_state**2 + dim_measurement**2),
            output_dim=dim_state
            * dim_measurement,  # Kalman gain dimension is (dim_state x dim_measurement)
            hidden_dim=dim_state * dim_measurement,
            layers=3,
        )

        # Reset the hidden state to 1 batch size
        self.reset()

    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """Reset the hidden state of the GRU cell.

        Args:
            batch_size (int): The batch size of the input tensor. Default is 1.

        Returns:
            torch.Tensor: The reset hidden state tensor.
        """
        self.hx = torch.zeros(
            self._gru_layers,
            batch_size,
            self.hx_dim,
            device=self.device,
            dtype=self.dtype,
        )

    def to(self, *args, **kwargs):
        """Move the KalmanNet to a specific device.

        Args:
            *args: Variable length argument list.
        """
        super().to(*args, **kwargs)
        self.hx = self.hx.to(*args, **kwargs)
        return self

    def _sanity_check(self, combinations: dict[str, torch.Tensor]) -> None:
        """Sanity check for the input dictionary.

        Args:
            combinations (dict[str, torch.Tensor]): The dictionary of combinations of the input tensors. [flavour: tensor , hx: tensor]
        """
        # Check if the dictionary is not empty
        if not combinations:
            raise ValueError("Empty dictionary of combinations")

        # Check if the flavours are present
        for flavour in self._flavours:
            if flavour not in combinations:
                raise ValueError(f"Missing flavour: {flavour}")

        # Check if the flavours have the correct shape
        for flavour in self._flavours:
            if (
                combinations[flavour].shape[-1] != self.flavour_dict[flavour]
                or combinations[flavour].shape[0] != self.hx.shape[1]
            ):
                raise ValueError(
                    f"""Invalid flavour shape! The shape of {flavour} must be ({self.hx.shape[1]}, {self.flavour_dict[flavour]}). Got {combinations[flavour].shape} instead.
                    If this is the first iteration, the batch size of the input tensor must be the same as the batch size of the hidden state tensor. Use the 
                    reset() method to reset the hidden state tensor to the desired batch size.
                    """
                )

    def forward(self, combinations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the KalmanNet.

        Args:
            combinations (dict[str, torch.Tensor]): The dictionary of combinations of the input tensors. [flavour: tensor]

        Note:
            All the flavours must be have appropriate last dimension to be stacked together. The F1, F2 flavours must have the
            shape of (batch_size, dim_measurement) and the F3, F4 flavours must have the shape of (batch_size, dim_state).


        Returns:
            torch.Tensor: The kalman gain tensor. Shape (batch_size, dim_state, dim_measurement)
        """
        # Sanity check for the input dictionary
        self._sanity_check(combinations)

        # Order the dictionary by keys
        # Essentially, this is a sanity check to ensure that the input dictionary is ordered in sucessive runs
        combinations = dict(sorted(combinations.items(), key=lambda item: item[0]))

        # Normalize the combinations
        for key in combinations:
            combinations[key] = torch.nn.functional.normalize(
                combinations[key],
                p=2,
                dim=-1,
                eps=1e-12,
            )

        # Concat the input tensors to form the input tensor
        # Shape: (batch_size , sum([flavour_dict[flavour] for flavour in self._flavours]))
        x = torch.cat(list(combinations.values()), dim=-1)

        # Add a temporal dimension to the input tensor
        x = x.unsqueeze(0)

        # Pass through the input layer
        x = self.input_layer(x)
        # Pass through the GRU cell
        # Where Q is the joint covariance of shape (batch_size, hx_dim)
        # and hx is the hidden state of shape (layers, batch_size, hx_dim)
        Q, self.hx = self.network(x, self.hx)

        # Squeeze the temporal dimension of Q which has the shape (1, batch_size, hx_dim)
        Q = Q.squeeze(0)  # Shape: (batch_size, hx_dim)
        # Pass through the output layer
        x = self.output_layer(Q)  # Shape: (batch_size, dim_state * dim_measurement)

        return x.view(-1, self._dim_state, self._dim_measurement)

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
