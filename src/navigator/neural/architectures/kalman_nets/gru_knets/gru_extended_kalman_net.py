"""Implements the original version of the Extended KalmanNet using GRU cells.

This module provides the original implementation of the Extended KalmanNet using GRU cells. Instead of jointly tracking
the state and measurement covariance, this version individually tracks the state and measurement covariance. The Kalman gain
is then computed using the individual covariances.

Following are tracked by the GRU cells:
    - The state covariance
    - The measurement covariance
    - The process noise covariance

For more details, refer to the paper:
- G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun, and Y. C. Eldar,
  "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics,"
  in IEEE Transactions on Signal Processing, vol. 70, pp. 1532-1547, 2022, doi: 10.1109/TSP.2022.3158588.

Source:
    - G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun, and Y. C. Eldar,
      "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics,"
      in IEEE Transactions on Signal Processing, vol. 70, pp. 1532-1547, 2022, doi: 10.1109/TSP.2022.3158588.
"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from .linear_blocks import LinearBlocks, SimpleLinearBlock


class GRUExtendedKalmanBlock(LightningModule):
    """The original Extended KalmanNet implementation using GRU cells.

    This class implements the Extended KalmanNet, a variant of the Kalman filter, using GRU cells. It provides an
    original approach where the state covariance, measurement covariance, and process noise covariance are individually
    tracked by GRU cells. The Kalman gain is then computed using these individual covariances.

    Note:
        - To use the Extended KalmanNet, all the flavors must be provided. The flavors are the types of inputs
          the KalmanNet has been trained on.

    Args:
        dim_state (int): Dimension of the state space.
        dim_measurement (int): Dimension of the measurement space.
        flavours (list[str]): List of flavors for the KalmanNet.
        **kwargs: Additional keyword arguments for the Extended KalmanNet.

    Raises:
        ValueError: If invalid dimensions or flavors are provided.

    Properties:
        Q_dim (int): Dimension of the process noise.
        R_dim (int): Dimension of the measurement noise.
        S_dim (int): Dimension of the measurement covariance.
        P_dim (int): Dimension of the state covariance.
    """

    def __init__(
        self,
        dim_state: int,
        dim_measurement: int,
        gru_layers: int = 1,
        state_expansion_factor: int = 2,
        measurement_expansion_factor: int = 2,
        gain_layers: int = 1,
        layer_norm: bool = False,
        **kwargs,
    ) -> None:
        """The extended Kalman Net class for the KalmanNet.

        Args:
            dim_state (int): The dimension of the state space.
            dim_measurement (int): The dimension of the measurement space.
            gru_layers (int): The number of GRU layers. Default is 1.
            state_expansion_factor (int): The expansion factor for the hidden dimensions of the state before the GRU cells.
            measurement_expansion_factor (int): The expansion factor for the hidden dimensions of the measurement before the GRU cells.
            gain_layers (int): The number of linear layers for processing the outputs of the GRU cells to kalman gain.
            layer_norm (bool): Whether to use layer normalization. Default is False.
            **kwargs: The keyword arguments for the extended Kalman Net.

        Note: The flavours are what the kalman net are trained on.
        """
        super().__init__()

        if dim_state <= 0:
            raise ValueError(f"Invalid dim_state: {dim_state}")
        if dim_measurement <= 0:
            raise ValueError(f"Invalid dim_measurement: {dim_measurement}")

        # Store the input parameters
        self.dim_state = dim_state
        self.dim_measurement = dim_measurement
        self.gru_layers = gru_layers
        self.layer_norm = layer_norm

        # Flavours dictionary
        self.flavour_dict = {
            "F1": dim_measurement,
            "F2": dim_measurement,
            "F3": dim_state,
            "F4": dim_state,
        }

        # Initialize the intialize each module by the input parameters
        self.networks = nn.ModuleDict(
            {
                "Q_LINEAR": SimpleLinearBlock(
                    input_dim=self.dim_state,
                    output_dim=self.dim_state * state_expansion_factor,
                ),
                "Q_GRU": nn.GRU(
                    input_size=self.dim_state * state_expansion_factor,
                    hidden_size=self.Q_dim,
                    num_layers=self.gru_layers,
                ),
                "SIGMA_LINEAR": SimpleLinearBlock(
                    input_dim=self.dim_state,
                    output_dim=self.dim_state * state_expansion_factor,
                ),
                "SIGMA_GRU": nn.GRU(
                    input_size=self.Q_dim + self.dim_state * state_expansion_factor,
                    hidden_size=self.P_dim,
                    num_layers=self.gru_layers,
                ),
                "SIGMA_TO_S_EXPAND": SimpleLinearBlock(
                    input_dim=self.P_dim,
                    output_dim=self.S_dim,
                ),
                "S_GRU_LINEAR": SimpleLinearBlock(
                    input_dim=self.dim_measurement * 2,
                    output_dim=self.dim_measurement * 2 * measurement_expansion_factor,
                ),
                "S_GRU": nn.GRU(
                    input_size=self.S_dim
                    + self.dim_measurement * 2 * measurement_expansion_factor,
                    hidden_size=self.S_dim,
                    num_layers=self.gru_layers,
                ),
                "KG_LINEAR": LinearBlocks(
                    input_dim=self.S_dim + self.P_dim,
                    output_dim=self.dim_state * self.dim_measurement,
                    hidden_dim=self.S_dim + self.P_dim,
                    layers=gain_layers,
                    output_layer=True,
                ),
                "SIGMA_KG_COMBINE": SimpleLinearBlock(
                    input_dim=self.S_dim + self.dim_state * self.dim_measurement,
                    output_dim=self.P_dim,
                ),
                "P_UPDATE": SimpleLinearBlock(
                    input_dim=self.P_dim * 2,
                    output_dim=self.P_dim,
                ),
            }
        )

        # Reset the hidden states
        self.reset()

    def reset(self, batch_size: int = 1) -> None:
        """Reset the hidden state of the GRU cells.

        Args:
            batch_size (int): The batch size of the input tensor. Default is 1.
        """
        self.Q_GRU_HIDDEN = torch.zeros(
            self.gru_layers,
            batch_size,
            self.Q_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.SIGMA_GRU_HIDDEN = torch.zeros(
            self.gru_layers,
            batch_size,
            self.P_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.S_GRU_HIDDEN = torch.zeros(
            self.gru_layers,
            batch_size,
            self.S_dim,
            device=self.device,
            dtype=self.dtype,
        )

    def to(self, *args, **kwargs):
        """Move the extended Kalman Net to the specified device.

        This is needed to copy the internal state of the GRU cells to the new device.

        Args:
            *args: The positional arguments for the to method.
            **kwargs: The keyword arguments for the to method.
        """
        self = super().to(*args, **kwargs)
        self.Q_GRU_HIDDEN = self.Q_GRU_HIDDEN.to(*args, **kwargs)
        self.SIGMA_GRU_HIDDEN = self.SIGMA_GRU_HIDDEN.to(*args, **kwargs)
        self.S_GRU_HIDDEN = self.S_GRU_HIDDEN.to(*args, **kwargs)
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
        for flavour in self.flavour_dict:
            if flavour not in combinations:
                raise ValueError(f"Missing flavour: {flavour}")

        # Check if the flavours have the correct shape
        for flavour in self.flavour_dict:
            if (
                combinations[flavour].shape[-1] != self.flavour_dict[flavour]
                or combinations[flavour].shape[0] != self.Q_GRU_HIDDEN.shape[1]
            ):
                raise ValueError(
                    f"""Invalid flavour shape! The shape of {flavour} must be ({self.Q_GRU_HIDDEN.shape[1]}, {self.flavour_dict[flavour]}). Got {combinations[flavour].shape} instead.
                    If this is the first iteration, the batch size of the input tensor must be the same as the batch size of the hidden state tensor. Use the 
                    reset() method to reset the hidden state tensor to the desired batch size.
                    """
                )

    def forward(self, combinations: dict[str, torch.Tensor]) -> torch.Tensor:
        """The forward pass for the extended Kalman Net.

        Args:
            combinations (dict[str, torch.Tensor]): The dictionary of combinations of the input tensors.

        Input Format:
            The combination dictionary must contain the following keys iff the keys are present in the flavours:
                - "F1" : The F1 combination tensor. DIM: (batch, dim_measurement)
                - "F2" : The F2 combination tensor. DIM: (batch, dim_measurement)
                - "F3" : The F3 combination tensor. DIM: (batch, dim_state)
                - "F4" : The F4 combination tensor. DIM: (batch, dim_state)
        Returns:
            torch.Tensor: The Kalman Gain tensor. DIM: (batch, dim_state, dim_measurement)

        """
        # Sanity check for the input dictionary
        self._sanity_check(combinations)

        # Individually normalize the combinations if required
        if self.layer_norm:
            for key in combinations:
                combinations[key] = nn.functional.normalize(
                    combinations[key], dim=-1, p=2, eps=1e-12
                )

        #### Tracking Step of KalmanNet ####

        # Process Noise Covariance Tracking Step of KalmanNet
        # Shape: (1, batch, dim_state * expansion_factor)
        Q_INPUT = self.networks["Q_LINEAR"](combinations["F4"]).unsqueeze(0)
        # Shape: (1, batch, Q_dim) , (gru_layers, batch, Q_dim)
        Q, self.Q_GRU_HIDDEN = self.networks["Q_GRU"](Q_INPUT, self.Q_GRU_HIDDEN)

        # State Covariance Tracking Step of KalmanNet
        # Shape: (1, batch, dim_state * expansion_factor)
        SIGMA_INPUT = self.networks["SIGMA_LINEAR"](combinations["F3"]).unsqueeze(0)
        # State Covariance Tracking Step of KalmanNet
        # Shape: (1, batch, P_dim) , (gru_layers, batch, P_dim)
        SIGMA, self.SIGMA_GRU_HIDDEN = self.networks["SIGMA_GRU"](
            torch.cat((Q, SIGMA_INPUT), dim=-1), self.SIGMA_GRU_HIDDEN
        )

        # Measurement Covariance Tracking Step of KalmanNet
        # Expand the state covariance to the measurement covariance
        # Shape: (1, batch, S_dim)
        SIGMA_TO_S_EXPANSION = self.networks["SIGMA_TO_S_EXPAND"](SIGMA)
        # Shape: (1, batch,  2 * self.dim_measurement * expansion_factor)
        S_INPUT = self.networks["S_GRU_LINEAR"](
            torch.cat((combinations["F1"], combinations["F2"]), dim=-1)
        ).unsqueeze(0)
        # Shape: (1, batch, S_dim) , (gru_layers, batch, S_dim)
        S, self.S_GRU_HIDDEN = self.networks["S_GRU"](
            torch.cat((SIGMA_TO_S_EXPANSION, S_INPUT), dim=-1), self.S_GRU_HIDDEN
        )

        # Kalman Gain Step of KalmanNet
        # Shape: (batch, self.P_dim + self.S_dim)
        KG_INPUT = torch.cat((SIGMA, S), dim=-1)
        # Shape: (batch, dim_state * dim_measurement)
        KG = self.networks["KG_LINEAR"](KG_INPUT)

        ## Update Step of KalmanNet ##

        # Combine the measurement and state covariance to update the state covariance
        # Shape (batch , S + dim_state * dim_measurement)
        SIGMA_KG_COMBINE_INPUT = torch.cat((S, KG), dim=-1)
        # Shape (batch , P_dim)
        SIGMA_KG_COMBINE = self.networks["SIGMA_KG_COMBINE"](SIGMA_KG_COMBINE_INPUT)

        # Update the state covariance
        # Shape (batch , P_dim + P_dim)
        SIGMA_UPDATE_INPUT = torch.cat((SIGMA, SIGMA_KG_COMBINE), dim=-1)
        # Shape (batch , P_dim)
        SIGMA_UPDATE = self.networks["P_UPDATE"](SIGMA_UPDATE_INPUT)

        # Update the hidden state
        self.SIGMA_GRU_HIDDEN = SIGMA_UPDATE.repeat(
            self.gru_layers, 1, 1
        )  # Need to account for the gru layers

        # Return the Kalman Gain
        return KG.view(-1, self.dim_state, self.dim_measurement)

    @property
    def Q_dim(self) -> int:
        """The dimension of the process noise.

        Returns:
            int: The dimension of the process noise.
        """
        return self.dim_state**2

    @Q_dim.setter
    def Q_dim(self, value: int) -> None:  # noqa :ARG002
        """The setter for the dimension of the process noise.

        Args:
            value (int): The new dimension of the process noise.
        """
        raise AttributeError(
            "Use the constructor to set the dimension of the process noise."
        )

    @property
    def R_dim(self) -> int:
        """The dimension of the measurement noise.

        Returns:
            int: The dimension of the measurement noise.
        """
        return self.dim_measurement**2

    @R_dim.setter
    def R_dim(self, value: int):  # noqa :ARG002
        """The setter for the dimension of the measurement noise.

        Args:
            value (int): The new dimension of the measurement noise.
        """
        raise AttributeError(
            "Use the constructor to set the dimension of the measurement noise."
        )

    @property
    def S_dim(self) -> int:
        """The dimension of the measurement covariance.

        Returns:
            int: The dimension of the state covariance.
        """
        return self.dim_measurement**2

    @S_dim.setter
    def S_dim(self, value: int):  # noqa :ARG002
        """The setter for the dimension of the measurement covariance.

        Args:
            value (int): The new dimension of the measurement covariance.
        """
        raise AttributeError(
            "Use the constructor to set the dimension of the measurement covariance."
        )

    @property
    def P_dim(self) -> int:
        """The dimension of the state covariance.

        Returns:
            int: The dimension of the state covariance.
        """
        return self.dim_state**2

    @P_dim.setter
    def P_dim(self, value: int):  # noqa :ARG002
        """The setter for the dimension of the state covariance.

        Args:
            value (int): The new dimension of the state covariance.
        """
        raise AttributeError(
            "Use the constructor to set the dimension of the state covariance."
        )
