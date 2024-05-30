"""Module for KalmanNet filter implementations.

This module defines an abstract base class, AbstractKalmanNet, which serves as the foundation for various KalmanNet
implementations. KalmanNets are deep learning-accelerated Kalman filters that utilize neural networks to approximate the
state of a dynamic system. The package includes the KalmanNet and its variants, such as GRUKalmanNet, LSTMKalmanNet,
TransformerKalmanNet, etc.

Design Pattern:
    The AbstractKalmanNet class provides a structured foundation for KalmanNet and its variants. It inherits from the
    pytorch_lightning.LightningModule and ABC (Abstract Base Class), facilitating extensibility and modularity. The design
    pattern enables seamless integration as a drop-in replacement for traditional Kalman filters in existing codebases.

Source:
    - G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun, and Y. C. Eldar,
      "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics,"
      in IEEE Transactions on Signal Processing, vol. 70, pp. 1532-1547, 2022, doi: 10.1109/TSP.2022.3158588.

Usage:
    To use the provided KalmanNet implementations, instantiate the desired class (e.g., KalmanNet, GRUKalmanNet) and
    utilize the predict and update methods for filtering. Additionally, subclassing AbstractKalmanNet allows for the
    creation of custom KalmanNet filters tailored to specific applications.

Example:
    ```python
    from navigator.filters.kalman_nets import KalmanNet, GRUKalmanNet, LSTMKalmanNet, TransformerKalmanNet

    # Instantiate KalmanNet and its variants
    kalman_net_instance = KalmanNet(dim_state=..., dim_measurement=..., dt=...)
    gru_kalman_net_instance = GRUKalmanNet(dim_state=..., dim_measurement=..., dt=...)
    lstm_kalman_net_instance = LSTMKalmanNet(dim_state=..., dim_measurement=..., dt=...)
    transformer_kalman_net_instance = TransformerKalmanNet(dim_state=..., dim_measurement=..., dt=...)
    ```


Author:
    Nischal Bhattarai (nischalbhattaraipi@gmail.com)
"""

from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch

from ....utility.tracker import HistoryTracker


class AbstractKalmanNet(pl.LightningModule, ABC):
    """The Kalman Net abstract class.

    This class provides the base structure for the Kalman Net and its variants.


    Source:
        G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun and Y. C. Eldar,
        "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics,"
        in IEEE Transactions on Signal Processing, vol. 70, pp. 1532-1547, 2022, doi: 10.1109/TSP.2022.3158588.

    Convention:
        B - Batch Dimension
        T - Time or Trajectory Dimension
        D - Dimension of the state vector
        M - Dimension of the measurement vector
    """

    POSSIBLE_FLAVORS = ["F1", "F2", "F3", "F4"]  # See the paper for what these are!

    def __init__(
        self,
        dim_state: int,
        dim_measurement: int,
        dt: float,
        flavor: list[str] = ["F1", "F2", "F3", "F4"],
        max_history: int = 2,
        track_f2_loss: bool = False,
    ) -> None:
        """Initializes the Kalman Net.

        Args:
            dim_state (int): Dimension of the state vector.
            dim_measurement (int): Dimension of the measurement vector.
            dt (float): Time step for the filter.
            dtype (torch.dtype, optional): Data type for the filter. Defaults to torch.float32.
            flavor (list[str], optional): List of flavors. Defaults to ["F1", "F2", "F3", "F4"].
            max_history (int, optional): Maximum history to track. Defaults to 2.
            track_f2_loss (bool, optional): Whether to track the F2 loss. Defaults to False.

        Returns:
            None
        """
        # Call the super class constructor to initialize the Module
        super().__init__()

        # Check the dimension of the state vector
        if dim_state <= 0 or dim_measurement <= 0:
            raise ValueError(
                "The dimensions of the state and measurement vectors should be greater than 0."
            )

        # Set the dimension of the state vector
        self.dim_state = dim_state
        # Set the dimension of the measurement vector
        self.dim_measurement = dim_measurement

        # Set the time step
        if dt <= 0:
            raise ValueError("The time step should be greater than 0.")
        self.dt = dt

        # Check the flavors is valid
        if not all([flavor in self.POSSIBLE_FLAVORS for flavor in flavor]):
            raise ValueError(
                f"Invalid flavor! The possible flavors are {self.POSSIBLE_FLAVORS}"
            )
        self.flavor = flavor

        # Set the maximum history to track
        if max_history <= 1:
            raise ValueError("The maximum history should be greater than 1.")
        self.max_history = max_history

        # Set the batch dimension
        self.batch_dim = 1

        # See if F2 is enabled to track the robust error
        if "F2" not in self.flavor and track_f2_loss:
            raise ValueError("Cannot track robust error without F2 flavor.")
        self.track_f2_loss = track_f2_loss

        # Innovation parameters
        self.W_innov = torch.nn.Parameter(
            torch.randn(dim_measurement, dim_measurement), requires_grad=True
        )
        self.b_innov = torch.nn.Parameter(
            torch.randn(dim_measurement), requires_grad=True
        )

        # Set the internal state of the filter
        self.reset_trackers(batch_dim=self.batch_dim)

    def _tracker_to(self, tracker: HistoryTracker, *args, **kwargs) -> HistoryTracker:
        """Copies the internal state of the tracker to the device or dtype.

        Args:
            tracker (HistoryTracker): The tracker to copy the internal state of.
            *args: Arguments for the to method.
            **kwargs: Keyword arguments for the to method.

        Returns:
            HistoryTracker: The tracker with the internal state copied to the new device or dtype.
        """
        new_tracker = HistoryTracker(max_history=tracker.max_history)

        for item in tracker:
            new_tracker.add(item.to(*args, **kwargs))

        return new_tracker

    # Override to method to copy the internal state of the filter to the device or dtype
    def to(self, *args, **kwargs) -> "AbstractKalmanNet":
        """Copies the internal state of the filter to the device or dtype.

        This method is overridden to ensure that the internal state of the filter is copied to the device or dtype
        when the filter is moved to a new device or dtype.

        Args:
            *args: Arguments for the to method.
            **kwargs: Keyword arguments for the to method.

        Returns:
            AbstractKalmanNet: The filter with the internal state copied to the new device or dtype.
        """
        # Call the super class to method
        super().to(*args, **kwargs)

        # Replace the measurement and state trackers with new ones
        self.measuremnt_tracker = self._tracker_to(
            self.measuremnt_tracker, *args, **kwargs
        )
        self.state_tracker = self._tracker_to(self.state_tracker, *args, **kwargs)
        self.prior_state_tracker = self._tracker_to(
            self.prior_state_tracker, *args, **kwargs
        )
        self.f2_loss_tracker = self._tracker_to(self.f2_loss_tracker, *args, **kwargs)

        # Return the filter
        return self

    def reset_trackers(self, batch_dim: int = 1) -> None:
        """Resets the internal state of the filter for new predict-update cycle.

        Args:
            batch_dim (int, optional): Batch dimension of the state vector. Defaults to 1.

        Warning:
            This function is essential for the filter to work correctly in training and inference.
            Otherwise previous state and measurement vectors will be used in the next predict-update cycle.
            This might lead to unexpected behavior or exploding state vectors due to the accumulation of errors.

        Returns:
            None
        """
        # Store the current batch dimension
        self.batch_dim = batch_dim

        # Clear the measurement and state trackers
        self.measuremnt_tracker = HistoryTracker(max_history=self.max_history)
        self.state_tracker = HistoryTracker(max_history=self.max_history)
        self.prior_state_tracker = HistoryTracker(max_history=self.max_history)
        self.f2_loss_tracker = HistoryTracker(max_history=-1)

        # Add the posterior state vector to the state tracker
        self.state_tracker.add(
            torch.zeros(
                self.batch_dim,
                self.dim_state,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True if self.track_f2_loss else False,
            )
        )
        # Add the prior state vector to the state tracker
        self.measuremnt_tracker.add(
            torch.zeros(
                self.batch_dim,
                self.dim_measurement,
                device=self.device,
                dtype=self.dtype,
            )
        )

    @abstractmethod
    def hx(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Measurement function. This function is implemented by the child class.

        Args:
            x (torch.Tensor): Current state vector. Shape (dim_state,) 1D array.
            kwargs (dict[str, torch.Tensor]): Additional arguments for the measurement function. (e.g.sv_coords, etc.)

        Note:
            Any kwargs tensor should have the first index as batch dimension since this is evaluated along the batch dimension for each state vector.

        Returns:
            ndarray: Measurement vector. Shape (dim_measurement,)
        """
        pass

    @abstractmethod
    def fx(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """State transition function.

        Args:
            x (torch.Tensor): Current state vector. Shape (dim_state,) 1D array.
            kwargs (dict[str, torch.Tensor]): Additional arguments for the state transition function. (e.g. control input, etc.)

        Note:
            Any kwargs tensor should have the first index as batch dimension since this is evaluated along the batch dimension for each state vector.

        Returns:
            torch.Tensor: Predicted state vector. Shape (dim_state,)
        """
        pass

    @abstractmethod
    def calculate_kalman_gain(
        self, combinations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculates the Kalman gain.

        This function is implemented by the child class. The function should calculate the Kalman gain using the combinations.
        This combination is fed to the RNN or Transformer layer to calculate the Kalman gain which should have the dimension of
        (dim_state, dim_measurement).

        Note: The combinations have batch dimension as the first index. So any neural network layer should be able to handle
        the first index as batch dimension.

        Args:
            combinations (dict[str, torch.Tensor]): Dictionary of combinations.

        Returns:
            torch.Tensor: Kalman gain.
        """
        pass

    @abstractmethod
    def reset(self, batch_dim: int) -> None:
        """Resets the internal state of the filter to batch_dim.

        This method is implemented by the child class. This method should reset the internal state of the filter to batch_dim
        including all the state of associated tracker and neural networks.

        Args:
            batch_dim (int): Batch dimension of the state vector.

        Returns:
            None
        """
        pass

    def predict(self, fx_kwargs: dict[str, torch.Tensor] = {}) -> torch.Tensor:
        """Predicts the state vector at the next time step.

        Args:
            fx (callable): State transition function.
            fx_kwargs (dict[str, torch.Tensor]): Additional arguments for the state transition function.

        Returns:
            torch.Tensor: Predicted state vector. Shape (batch_size, dim_state).
        """
        # Call the state transition function to get the predicted state vector
        # Do a state transition along the batch dimension
        return torch.vstack(
            [
                self.fx(
                    self.state_tracker[-1][i],
                    **{key: value[i] for key, value in fx_kwargs.items()},
                )
                for i in range(self.batch_dim)
            ]
        )

    def _sanity_check(
        self,
        y: torch.Tensor,
        hx_kwargs: dict[str, torch.Tensor],
        fx_kwargs: dict[str, torch.Tensor],
    ) -> None:
        """Performs sanity check on the input to the update method.

        Args:
            y (torch.Tensor): Measurement vector. Shape (batch_size, dim_measurement).
            hx_kwargs (dict[str, torch.Tensor]): Additional arguments for the measurement function.
            fx_kwargs (dict[str, torch.Tensor]): Additional arguments for the state transition function.

        Raises:
            ValueError: If the measurement vector has an invalid shape.
        """
        # Check if the measurement has the proper shape
        if y.shape != (self.batch_dim, self.dim_measurement):
            raise ValueError(
                f"The measurement vector should have the shape {(self.batch_dim, self.dim_measurement)} but got {y.shape}."
            )

        # Check if the kwargs has the proper shape
        if not all([value.shape[0] == self.batch_dim for value in hx_kwargs.values()]):
            raise ValueError(
                "The kwargs should have the first index as batch dimension."
            )

        if not all([value.shape[0] == self.batch_dim for value in fx_kwargs.values()]):
            raise ValueError(
                "The kwargs should have the first index as batch dimension."
            )

    def predict_update(
        self,
        y: torch.Tensor,
        hx_kwargs: dict[str, torch.Tensor] = {},
        fx_kwargs: dict[str, torch.Tensor] = {},
    ) -> torch.Tensor:
        """Performs the predict-update cycle for the Kalman filter.

        Args:
            y (torch.Tensor): Measurement vector. Shape (batch_size, dim_measurement).
            hx_kwargs (dict[str, torch.Tensor]): Additional arguments for the measurement function.
            fx_kwargs (dict[str, torch.Tensor]): Additional arguments for the state transition function.

        Returns:
            torch.Tensor: Predicted state vector. Shape (batch_size, dim_state).

        Note:
            If the hx_kwargs or fx_kwargs is provided, it is assumend that each value in the dictionary has an index for
            each batch data. For example, if the measurement is of shape (3, 10), then for every key in the
            dictionary, the value should be of shape (3, ...). This is because the measurement function is
            evaluated along the batch dimension for each state vector.

        """
        # Sanity check
        self._sanity_check(y, hx_kwargs, fx_kwargs)

        # Calculate the prior state vector
        x_prior = self.predict(fx_kwargs=fx_kwargs)

        # Calculate the combinations
        combinations = {
            "F1": y
            - self.measuremnt_tracker[
                -1
            ],  # The difference between the current measurement and the previous measurement. Shape (batch_size, dim_measurement)
            "F2": y
            - torch.vstack(
                [
                    self.hx(
                        x_prior[i],
                        **{key: value[i] for key, value in hx_kwargs.items()},
                    )
                    for i in range(self.batch_dim)
                ]  # The difference between the current measurement and the predicted measurement. Shape (batch_size, dim_measurement)
            ),
            "F3": (
                self.state_tracker[-1] - self.state_tracker[-2]
                if self.state_tracker.is_full()  # The difference between the current state and the previous state calculated at t-1 for timestep t. Shape (batch_size, dim_state)
                else self.state_tracker[-1]
            ),
            "F4": (
                self.state_tracker[-1]
                - self.prior_state_tracker[
                    -1
                ]  # The difference between the previous posterior state and previous prior state calculated at t-1 for timestep t. Shape (batch_size, dim_state)
                if not self.prior_state_tracker.is_empty()
                else self.state_tracker[-1]
            ),
        }

        # Mask Combinations based on the flavor passed for the filter
        combinations = {
            key: value for key, value in combinations.items() if key in self.flavor
        }

        # Calculate the kalman gain
        kalman_gain = self.calculate_kalman_gain(
            combinations
        )  # Shape (batch_size, dim_state, dim_measurement)

        # Update the state vector
        x_posterior = x_prior + torch.einsum(
            "ijk,ik->ij",
            kalman_gain,
            torch.tanh(
                torch.einsum("jj,ij->ij", self.W_innov, combinations["F2"])
                + self.b_innov
            ),
        )

        # Update the states of the filter for next time step
        self.measuremnt_tracker.add(y)  # Add measurement to the measurement tracker
        self.state_tracker.add(
            x_posterior.detach() if self.track_f2_loss else x_posterior
        )  # Add the posterior state to the state tracker
        self.prior_state_tracker.add(
            x_prior
        )  # Add the prior state to the prior state tracker

        # If robust error tracker is enabled, then add the error to the tracker
        if self.track_f2_loss:
            self.f2_loss_tracker.add(combinations["F2"])

        return x_posterior

    def forward(
        self,
        x: torch.Tensor,
        hx_kwargs: dict[str, torch.Tensor] = {},
        fx_kwargs: dict[str, torch.Tensor] = {},
    ) -> torch.Tensor:
        """This methods is convenience method to process the whole time series data.

        This method expects the input to be of shape (num_time_steps, batch_size, dim_state) and processes the whole time series
        at one go to compute the state vector at each time step.

        Note: This method assumes that the any kwargs passed through hx_kwargs or fx_kwargs has the first index as time step
        and second index as batch dimension. This is because the measurement function and state transition function is evaluated
        pointwise along the time and batch dimension at every time step and for every trajectory of the batch data.


        Args:
            x (torch.Tensor): The measurement vector. Shape (num_time_steps, batch_size, dim_measurement).
            hx_kwargs (dict[str, torch.Tensor]): Additional arguments for the measurement function.
            fx_kwargs (dict[str, torch.Tensor]): Additional arguments for the state transition function.


        Returns:
            torch.Tensor: Predicted state vector. Shape (timestep, batch_size, dim_state).
        """
        # Reset the internal state of the filter to batch_dim
        self.reset(batch_dim=x.shape[1])

        # Get the number of time steps
        T = x.shape[0]

        # output
        output = []

        # Iterate over the time steps
        for i in range(T):
            output.append(
                self.predict_update(
                    x[i],
                    hx_kwargs=(
                        {key: value[i] for key, value in hx_kwargs.items()}
                        if hx_kwargs
                        else {}
                    ),
                    fx_kwargs=(
                        {key: value[i] for key, value in fx_kwargs.items()}
                        if fx_kwargs
                        else {}
                    ),
                )
            )

        return torch.stack(output, dim=0)

    @property
    def f2_loss(self) -> torch.Tensor:
        """Returns the robust error.

        Defined to be the L2 norm of the all the elements in the robust error tracker.

        Returns:
            torch.Tensor: The robust error tracker.
        """
        if not self.track_f2_loss or self.f2_loss_tracker.is_empty():
            raise ValueError("The F2 error tracker is not enabled or empty.")

        return sum([torch.norm(error) for error in self.f2_loss_tracker.get()])

    @f2_loss.setter
    def f2_loss(self, value: torch.Tensor) -> None:  # noqa: ARG002
        """Enables or disables the robust error tracker.

        Args:
            value (bool): True to enable the robust error tracker, False to disable.

        Returns:
            None
        """
        raise ValueError("The robust error cannot be set directly.")


# Path: src/navigator/filters/kalman_nets/kalman_net.py
