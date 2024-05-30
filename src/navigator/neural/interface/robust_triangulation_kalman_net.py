"""Implements the robust kalman net for the neural network aided navigation system.

This is a varient of the KalmanNet[1] which uses self-attention mechanism to further increase the robustness of the Kalman Filter.
The varient has two main components, the first component is the KalmanNet[1] which is used to predict the state of the system  using GRU
Cell. However, the inputs to the GRU cells are further processed using the self-attention mechanism. The consecutive innovations are tracked
using a window of size `n` and the attention mechanism is used to give more weight to the innovations that are more recent and learn to recognize
the patterns in the innovations.
s

Triangulation Problem Formulation:
    A state of the system is defined as `x = [x, y, z, x_dot, y_dot, z_dot]` where `x, y, z` are the position of the system and `x_dot, y_dot, z_dot`
    are the velocity of the system. The measurement of the system is defined as range measurements  plus possibly additional measurements such as
    phase,doppler etc. The goal of the system is to estimate the state of the system given the measurements.

Classes:
    RobustTriangulationKalmanNet: Implements the robust kalman net for the neural network aided navigation system.

Author: 
    Nischal Bhattarai

Email:
    nischalbhattaraipi@gmail.com
"""

import numpy as np
import torch.nn as nn

from ..architectures.kalman_nets.kalman_net_base import AbstractKalmanNet
from ..architectures.set_transformer.set_attention_blocks import PMA, SAB


class RobustTriangulationKalmanNet(AbstractKalmanNet):
    """Abstract class for purposed the Robust Kalman Net."""

    def __init__(
        self,
        dim_state: int,
        dim_measurement: int,
        dt: float,
        history_window: int = 10,
        layer_norm: bool = False,
        sab_blocks: int = 3,
        set_transfomer_latent_dim: int = 16,
        rFF_dim: int = 16,
        innovation_latent_dim: int = 16,
    ) -> None:
        """Initializes the RobustKalmanNet.

        Args:
            dim_state: The dimension of the state of the system.
            dim_measurement: The dimension of the measurement of the system.
            dt: The time step of the system.
            history_window: The window size for the tracking the innovations. Default is 10.
            layer_norm: If True, the layer normalization is applied to the GRU cells. Default is False.
            set_transfomer_latent_dim: The latent dimension of the set transformer that converts tracker coordinates to the latent space. (Choose power of 2 for the latent dimension).
            sab_blocks: The number of set-attention blocks in the set-attention mechanism. Default is 3.
            rFF_dim: The dimension of the feed forward network in the self-attention mechanism. Default is 16.
            innovation_latent_dim: The latent dimension of the innovation history attention mechanism. (Choose greater that measurement dimension). Default is 16. (Choose power of 2 for the latent dimension).
        """
        super().__init__(
            dim_state=dim_state,
            dim_measurement=dim_measurement,
            dt=dt,
            flavor=["F1", "F2", "F3", "F4"],
            max_history=history_window,
            track_f2_loss=False,
        )
        # Initialize the parameters
        self.history_window = history_window
        self.layer_norm = layer_norm
        self.rFF_dim = rFF_dim

        # Choose power of 2 greater than 4 for the set transformer latent dimension.
        if (
            set_transfomer_latent_dim < 4
            or np.log2(set_transfomer_latent_dim).is_integer() is False
        ):
            raise ValueError(
                "The set transformer latent dimension must be greater than 4 and power of 2."
            )
        self.set_transfomer_latent_dim = set_transfomer_latent_dim

        # Initialize the set-transformer to transforme the given coordinate of the trackers like satellite coordinates to the latent space for RNN processing.
        # NOTE: Edit this as per the complexity of the problem.
        self.coordinate_set_transfomer = nn.Sequential(
            nn.Linear(
                3, self.set_transfomer_latent_dim
            ),  # (B,N,3)  -> (B,N, set_transfomer_latent_dim), convert the 3D coordinates to the latent space.
            *[
                SAB(
                    dim_Q=self.set_transfomer_latent_dim,
                    num_heads=4,
                    ln=self.layer_norm,
                    fnn_dim=self.rFF_dim,
                )
                for _ in range(sab_blocks)
            ],  # (B,  N, set_transfomer_latent_dim) -> (B,  N,  set_transfomer_latent_dim
            PMA(
                dim_Q=self.set_transfomer_latent_dim,
                dim_S=1,
                num_heads=4,
                ln=self.layer_norm,
                fnn_dim=self.rFF_dim,
            ),  # (B,  N,set_transfomer_latent_dim) -> (B,  1,  set_transfomer_latent_dim
            nn.Linear(
                self.set_transfomer_latent_dim, self.set_transfomer_latent_dim
            ),  # (B,  1,  set_transfomer_latent_dim) -> (B,  1,  set_transfomer_latent_dim
        )

        # Initialize the Innovation History Attention Mechanism
        if innovation_latent_dim < self.dim_measurement or (
            np.log2(innovation_latent_dim).is_integer() is False
            or innovation_latent_dim < 4
        ):
            raise ValueError(
                "The innovation latent dimension must be greater than the measurement dimension and power of 2 greater than 4."
            )

        self.attention_expansion_layer = nn.Linear(
            self.dim_measurement, innovation_latent_dim
        )
        self.innovation_attention = nn.MultiheadAttention(
            embed_dim=innovation_latent_dim,
            num_heads=4,
        )

        # Define the GRU Blocks for the Kalman Filter
        pass
