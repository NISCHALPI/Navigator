"""Set Transformer Based Continuous Satellite Tracker Module.

This module implements a continuous satellite tracker using a Set Transformer model. The tracker is designed for the Kalman Network and can handle jumps in the number of satellites being tracked. It maps satellite coordinates to a fixed-size latent set and feeds it to KalmanNet for Kalman filtering.

Classes:
    - SatelliteTracker: A Set Transformer based Continuous Satellite Tracker for Kalman Network.

Author:
    Nischal Bhattarai (nischalbhattaraipi@gmail.com)

"""

import torch
import torch.nn as nn

from ...architectures.set_transformer.set_attention_blocks import PMA, SAB

__all__ = ["SatelliteTracker"]


class SatelliteTracker(nn.Module):
    """A Set Transformer based Continuous Satellite Tracker for Kalman Network.

    This class implements a continuous satellite tracker using a Set Transformer model. It is designed to handle jumps in the number of satellites being tracked by mapping satellite coordinates to a fixed-size latent set and feeding it to KalmanNet for Kalman filtering.

    Attributes:
        SV_DIM (int): The dimension of satellite coordinates (default: 3).

    Args:
        num_sv (int): The number of satellites being continuously tracked.
        projection_dim (int): The dimension of the initial projection of the input satellite coordinates (default: 3).
        num_head (int): The number of attention heads in the Set Transformer model (default: 1).
        ln (bool): Whether to use layer normalization in the Set Transformer model (default: True).

    Raises:
        ValueError: If the num_head does not divide the projection_dim or if num_sv is less than 1.

    """

    SV_DIM = 3

    def __init__(
        self, num_sv: int, projection_dim: int = 3, num_head: int = 1, ln: bool = True
    ) -> None:
        """Initialize the Satellite Tracker.

        Args:
            num_sv (int): The number of satellites being continuously tracked.
            projection_dim (int): The dimension of the initial projection of the input satellite coordinates. Defaults to 3.
            num_head (int): The number of attention heads in the Set Transformer model. Defaults to 1.
            ln (bool, optional): Whether to use layer normalization in the Set Transformer model. Defaults to True.

        Raises:
            ValueError: if the num_head does not divide the projection_dim since this is fed to the Set Transformer model.

        Returns:
            None
        """
        super().__init__()
        if projection_dim % num_head != 0:
            raise ValueError(
                "The projection dimension must be divisible by the number of attention heads."
            )

        if num_sv < 1:
            raise ValueError("The number of satellites must be greater than 0.")

        self.num_sv = num_sv
        self.projection_dim = projection_dim
        self.num_head = num_head
        self.ln = ln

        # Set up the Encoder Transformer
        self.encoder = nn.Sequential(
            nn.Linear(self.SV_DIM, self.projection_dim),  # Initial Projection Layer
            SAB(
                dim_Q=self.projection_dim,
                num_heads=self.num_head,
                ln=self.ln,
                fnn_dim=self.projection_dim,
            ),
            SAB(
                dim_Q=self.projection_dim,
                num_heads=self.num_head,
                ln=self.ln,
                fnn_dim=self.projection_dim,
            ),
        )

        # Set up the Decoder Transformer
        self.decoder = nn.Sequential(
            PMA(
                dim_Q=self.projection_dim,
                dim_S=self.num_sv,  # The number of satellites being continuously tracked
                num_heads=1,
                ln=self.ln,
                fnn_dim=self.projection_dim,
            ),  # Pooling Layer to get the set representation return (-1, dim_S, projection_dim)
            SAB(
                dim_Q=self.projection_dim,
                num_heads=1,
                ln=self.ln,
                fnn_dim=self.projection_dim,
            ),
        )
        # Set up final linear layer to get the final output
        self.rFF = nn.Linear(self.projection_dim, self.SV_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Set Transformer based Continuous Satellite Tracker.

        Args:
            x (torch.Tensor): The input satellite coordinates of shape (B, N, 3)

        Returns:
            torch.Tensor: The transformed satellite coordinates of shape (B, num_sv, 3)
        """
        return self.rFF(self.decoder(self.encoder(x)))
