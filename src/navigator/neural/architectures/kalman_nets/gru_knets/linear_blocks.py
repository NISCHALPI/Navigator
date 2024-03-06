"""Implementation of the linear blocks for the KalmanNet."""

import torch
import torch.nn as nn

__all__ = ["LinearBlocks"]


class LinearBlocks(nn.Module):
    """The simple linear blocks.

    Args:
        nn (_type_): Base class for all neural network modules in PyTorch.
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int, layers: int
    ) -> None:
        """The linear blocks for the KalmanNet.

        Args:
            input_dim (int): The input dimension of the linear block.
            output_dim (int): The output dimension of the linear block.
            hidden_dim (int): The hidden dimension of the linear block.
            layers (int): The number of layers in the linear block.
        """
        super().__init__()

        # Check the validity of the parameters
        if input_dim <= 0:
            raise ValueError(f"Invalid input_dim: {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"Invalid output_dim: {output_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"Invalid hidden_dim: {hidden_dim}")
        if layers < 2:
            raise ValueError(f"Invalid layers: {layers} (should be at least 2)")

        # Initialize the network
        self.network = nn.ModuleList()

        # Add the input layer
        self.network.append(nn.Linear(input_dim, hidden_dim))
        self.network.append(nn.PReLU())
        # Add the hidden layers
        for _ in range(layers - 1):
            self.network.append(nn.Linear(hidden_dim, hidden_dim))
            self.network.append(nn.PReLU())
        # Add the output layer
        self.network.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the linear block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.network:
            x = layer(x)
        return x
