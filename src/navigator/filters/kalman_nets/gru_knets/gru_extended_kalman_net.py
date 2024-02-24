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

from .linear_blocks import LinearBlocks


class GRUExtendedKalmanBlock(nn.Module):
    """The original Extended KalmanNet implementation using GRU cells.

    This class implements the Extended KalmanNet, a variant of the Kalman filter, using GRU cells. It provides an
    original approach where the state covariance, measurement covariance, and process noise covariance are individually
    tracked by GRU cells. The Kalman gain is then computed using these individual covariances.

    Note:
        - To use the Extended KalmanNet, all the flavors must be provided. The flavors are the types of inputs
          the KalmanNet has been trained on.
        - The hidden states must be initialized by the user and have the following shapes:
            - Q_HIDDEN_KEY : Hidden state tensor for the process noise covariance GRU cell. DIM: (batch, Q_dim)
            - P_HIDDEN_KEY : Hidden state tensor for the state covariance GRU cell. DIM: (batch, P_dim)
            - S_HIDDEN_KEY : Hidden state tensor for the measurement covariance GRU cell. DIM: (batch, S_dim)

    Args:
        dim_state (int): Dimension of the state space.
        dim_measurement (int): Dimension of the measurement space.
        flavours (list[str]): List of flavors for the KalmanNet.
        **kwargs: Additional keyword arguments for the Extended KalmanNet.

    Attributes:
        Q_HIDDEN_KEY (str): Key for the hidden state tensor related to process noise covariance GRU cell.
        P_HIDDEN_KEY (str): Key for the hidden state tensor related to state covariance GRU cell.
        S_HIDDEN_KEY (str): Key for the hidden state tensor related to measurement covariance GRU cell.

    Raises:
        ValueError: If invalid dimensions or flavors are provided.

    Properties:
        Q_dim (int): Dimension of the process noise.
        R_dim (int): Dimension of the measurement noise.
        S_dim (int): Dimension of the measurement covariance.
        P_dim (int): Dimension of the state covariance.
    """

    Q_HIDDEN_KEY = "Q_GRU_HIDDEN"
    P_HIDDEN_KEY = "P_GRU_HIDDEN"
    S_HIDDEN_KEY = "S_GRU_HIDDEN"

    FLAVOUR_DICT = ["F1", "F2", "F3", "F4"]

    def __init__(
        self,
        dim_state: int,
        dim_measurement: int,
        flavours: list[str],
        **kwargs,
    ) -> None:
        """The extended Kalman Net class for the KalmanNet.

        Args:
            dim_state (int): The dimension of the state space.
            dim_measurement (int): The dimension of the measurement space.
            flavours (list[str]): The list of flavours for the KalmanNet. See the paper for more details.
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

        # Check the full flavours are provided
        if (
            not all([flavour in self.FLAVOUR_DICT for flavour in flavours])
            or len(flavours) != 4
        ):
            raise ValueError(
                f"To use the extended Kalman Net, all the flavours must be provided. The flavours provided are: {flavours}"
            )

        # Initialize the intialize each module by the input parameters
        self.networks = nn.ModuleDict(
            {
                "Q_GRU_INPUT": LinearBlocks(
                    input_dim=self.dim_state,
                    output_dim=kwargs.get("q_input_dim", self.dim_state * 2),
                    hidden_dim=kwargs.get("q_input_dim", self.dim_state * 2),
                    layers=kwargs.get("q_layers", 4),
                ),
                "Q_GRU_CELL": nn.GRUCell(
                    input_size=kwargs.get("q_input_dim", self.dim_state * 2),
                    hidden_size=self.Q_dim,
                ),
                "P_GRU_INPUT": LinearBlocks(
                    input_dim=self.dim_state,
                    output_dim=self.P_dim,
                    hidden_dim=kwargs.get("p_input_dim", self.dim_state * 2),
                    layers=kwargs.get("p_layers", 4),
                ),
                "P_GRU_CELL": nn.GRUCell(
                    input_size=self.Q_dim + self.P_dim,
                    hidden_size=self.P_dim,
                ),
                "P_TO_S_EXPAND": LinearBlocks(
                    input_dim=self.P_dim,
                    output_dim=self.S_dim,
                    hidden_dim=self.S_dim,
                    layers=kwargs.get("p_to_s_layers", 2),
                ),
                "S_GRU_INPUT": LinearBlocks(
                    input_dim=self.dim_measurement * 2,
                    output_dim=self.S_dim,
                    hidden_dim=kwargs.get("s_input_dim", self.dim_measurement * 2),
                    layers=kwargs.get("s_layers", 4),
                ),
                "S_GRU_CELL": nn.GRUCell(
                    input_size=self.S_dim * 2,
                    hidden_size=self.S_dim,
                ),
                "KG_LINEAR": LinearBlocks(
                    input_dim=self.S_dim + self.P_dim,
                    output_dim=self.dim_state * self.dim_measurement,
                    hidden_dim=self.dim_state * self.dim_measurement,
                    layers=kwargs.get("kg_layers", 4),
                ),
                "SK_COMBINE": LinearBlocks(
                    input_dim=self.S_dim + self.dim_state * self.dim_measurement,
                    output_dim=self.P_dim,
                    hidden_dim=self.P_dim,
                    layers=kwargs.get("sk_layers", 4),
                ),
                "P_UPDATE": LinearBlocks(
                    input_dim=self.P_dim * 2,
                    output_dim=self.P_dim,
                    hidden_dim=self.P_dim,
                    layers=kwargs.get("p_update_layers", 4),
                ),
            }
        )

    def forward(self, combinations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The forward pass for the extended Kalman Net.

        Args:
            combinations (dict[str, torch.Tensor]): The dictionary of combinations of the input tensors.

        Input Format:
        The combination dictionary must contain the following keys:
            - "F1" : The F1 combination tensor. DIM: (batch, dim_measurement)
            - "F2" : The F2 combination tensor. DIM: (batch, dim_measurement)
            - "F3" : The F3 combination tensor. DIM: (batch, dim_state)
            - "F4" : The F4 combination tensor. DIM: (batch, dim_state)
            - self.Q_HIDDEN_KEY : The hidden state tensor for the process noise covariance GRU cell. DIM: (batch, Q_dim)
            - self.P_HIDDEN_KEY : The hidden state tensor for the state covariance GRU cell. DIM: (batch, P_dim)
            - self.S_HIDDEN_KEY : The hidden state tensor for the measurement covariance GRU cell. DIM: (batch, S_dim)

            The name for these hidden key are defined as class variables to avoid any typos by the user.

        Output Format:
        The output dictionary contains the following keys:
            - "KG" : The Kalman Gain tensor. DIM: (batch, dim_state, dim_measurement)
            - self.Q_HIDDEN_KEY : The updated hidden state tensor for the process noise covariance GRU cell. DIM: (batch, Q_dim)
            - self.P_HIDDEN_KEY : The updated hidden state tensor for the state covariance GRU cell. DIM: (batch, P_dim)
            - self.S_HIDDEN_KEY : The updated hidden state tensor for the measurement covariance GRU cell. DIM: (batch, S_dim)

            The name for these hidden key are defined as class variables to avoid any typos by the user.


        Returns:
            dict[str, torch.Tensor]: The dictionary of the updated hidden states and the updated state covariance and kalman gain.


        """
        # Calculate the process noise covariance
        Q_inp = self.networks["Q_GRU_INPUT"](combinations["F4"])
        Q = self.networks["Q_GRU_CELL"](Q_inp, combinations[self.Q_HIDDEN_KEY])

        # Calculate the state covariance matrix
        P_inp = self.networks["P_GRU_INPUT"](combinations["F3"])
        P = self.networks["P_GRU_CELL"](
            torch.hstack([Q, P_inp]), combinations[self.P_HIDDEN_KEY]
        )

        # Calculate the measurement covariance matrix
        S_inp = self.networks["S_GRU_INPUT"](
            torch.hstack([combinations["F1"], combinations["F2"]])
        )
        S = self.networks["S_GRU_CELL"](
            torch.hstack([S_inp, self.networks["P_TO_S_EXPAND"](P)]),
            combinations[self.S_HIDDEN_KEY],
        )

        # Compute the Kalman Gain matrix
        KG = self.networks["KG_LINEAR"](
            torch.hstack([S, P]),
        )

        # Update the state covariance matrix P using the Kalman Gain and the measurement covariance
        P = self.networks["P_UPDATE"](
            torch.hstack([P, self.networks["SK_COMBINE"](torch.hstack([S, KG]))])
        )

        # Return the updated hidden states and the updated state covariance and kalman gain
        return {
            "KG": KG.view(-1, self.dim_state, self.dim_measurement),
            "Q_GRU_HIDDEN": Q,
            "P_GRU_HIDDEN": P,
            "S_GRU_HIDDEN": S,
        }

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
