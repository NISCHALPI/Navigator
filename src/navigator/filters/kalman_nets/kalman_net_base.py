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

from ...utility.tracker import HistoryTracker


class AbstractKalmanNet(pl.LightningModule, ABC):
    """The Kalman Net abstract class.

    This class provides the base structure for the Kalman Net and its variants.


    Source:
        G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun and Y. C. Eldar,
        "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics,"
        in IEEE Transactions on Signal Processing, vol. 70, pp. 1532-1547, 2022, doi: 10.1109/TSP.2022.3158588.

    Additonal Docs Here.

    """

    POSSIBLE_FLAVORS = ["F1", "F2", "F3", "F4"]  # See the paper for what these are!

    def __init__(
        self,
        dim_state: int,
        dim_measurement: int,
        dt: float,
        flavor: list[str] = ["F1", "F2", "F3", "F4"],
        max_history: int = 2,
    ) -> None:
        """Initializes the Kalman Net.

        Args:
            dim_state (int): Dimension of the state vector.
            dim_measurement (int): Dimension of the measurement vector.
            dt (float): Time step for the filter.
            flavor (list[str], optional): List of flavors. Defaults to ["F1", "F2", "F3", "F4"].
            max_history (int, optional): Maximum history to track. Defaults to 2.

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

        # Set the internal state of the filter
        self.reset_internal_state(batch_dim=self.batch_dim)

    def reset_internal_state(self, batch_dim: int = 1) -> None:
        """Resets the internal state of the filter for new predict-update cycle.

        This is called at the beginning of each tracking loop to reset the internal state of the filter. Note that
        this is very useful for the filter to train the filter in a batch way.

        The batch dim is the batch dimension of the state vector. For eg, if one is tracking position and velocity
        of a single object, the state vector will be of (x, vx) but if one wants to train it using batch data
        the state vector will be of (batch_size, state_dim) where batch_size is the number of trajectories being tracked.

        In the prediction loop, one can reset the internal state of the filter by calling this function with the
        batch_dim set one which tracks the state of the filter for each trajectory one temporal trajectory at a time.
        This reduces the filter to standard Kalman filter. However, if one wants to train the filter in a batch way,
        the batch_dim should be set to the number of trajectories being tracked.

        Note:
            The measurement function and the state transition function should have to account for the batch dimension
            since the state will be of shape (batch_size, state_dim) which is passed to the measurement and state transition
            functions.

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

        # Set the posterior state vector
        self.x_posterior = torch.zeros(
            batch_dim,
            self.dim_state,
            device=self.device,
            dtype=self.dtype,  # Ensure that the device and dtype are set
        )
        # Set the prior state vector
        self.x_prior = torch.zeros(
            batch_dim, self.dim_state, device=self.device, dtype=self.dtype
        )  # Ensure that the device and dtype are set

        # Clear the measurement and state trackers
        self.measuremnt_tracker = HistoryTracker(max_history=self.max_history)
        self.state_tracker = HistoryTracker(max_history=self.max_history)

        # Add the posterior state vector to the state tracker
        self.state_tracker.add(self.x_posterior)

    @abstractmethod
    def hx(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Measurement function. This function is implemented by the child class.

        Args:
            x (torch.Tensor): Current state vector. Shape (dim_state,) 1D array.
            kwargs (dict[str, torch.Tensor]): Additional arguments for the measurement function.


        Returns:
            ndarray: Measurement vector. Shape (batch_size, dim_measurement)
        """
        pass

    @abstractmethod
    def fx(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """State transition function.

        Args:
            x (torch.Tensor): Current state vector. Shape (dim_state,) 1D array.
            kwargs (dict[str, torch.Tensor]): Additional arguments for the state transition function.

        Returns:
            torch.Tensor: Predicted state vector. Shape (batch_size, dim_state).
        """
        pass

    @staticmethod
    def F1(y_curr: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        """Calculates the F1 combination (observation difference) for feeding to RNN layer.

        This combination captures the difference between the current measurement and the previous measurement.

        Args:
            y_curr (torch.Tensor): Current measurement vector. Shape (dim_measurement,) 1D array.
            y_prev (torch.Tensor): Previous measurement vector. Shape (dim_measurement,) 1D array.

        Returns:
            torch.Tensor: F1 combination. (dim_measurement)
        """
        return y_curr - y_prev

    @staticmethod
    def F2(y_curr: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """This calculates the F2 combination (innovation difference) for feeding to RNN layer.

        This combination captures the difference between the current measurement and the predicted measurement.

        Args:
            y_curr (torch.Tensor): Current measurement vector. Shape (batch_size, dim_state).
            y_pred (torch.Tensor): Predicted measurement vector. Shape (batch_size, dim_state).

        Returns:
            torch.Tensor: F2 combination. (dim_measurement)
        """
        return y_curr - y_pred

    @staticmethod
    def F3(x: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        """This calculates the F3 combination (forward evolution difference) for feeding to RNN layer.

        This combination encapsulates the state evolution process since this uses the state vectors of two consecutive posterior
        state vectors. This is taken for t-1 for time step t.

        Args:
            x (torch.Tensor): Current state vector. Shape (batch_size, dim_state).
            x_prev (torch.Tensor): Previous state vector. Shape (batch_size, dim_state).

        Returns:
            torch.Tensor: F3 combination. (dim_state)
        """
        return x - x_prev

    @staticmethod
    def F4(x_prior: torch.Tensor, x_posterior: torch.Tensor) -> torch.Tensor:
        """This calculates the F4 combination (forward update difference) for feeding to RNN layer.

        This combination encapsulates the state evolution process since this uses the state vectors of
        posterior and prior state vectors.

        Args:
            x_prior (torch.Tensor): Prior state vector. Shape (batch_size, dim_state).
            x_posterior (torch.Tensor): Posterior state vector. Shape (batch_size, dim_state).

        Returns:
            ndarray: F4 combination. (dim_state)
        """
        return x_prior - x_posterior

    @abstractmethod
    def calculate_kalman_gain(combinations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculates the Kalman gain.

        This function is implemented by the child class. The function should calculate the Kalman gain using the combinations.
        This combination is fed to the RNN or Transformer layer to calculate the Kalman gain which should have the dimension of
        (dim_state, dim_measurement).


        Args:
            combinations (dict[str, torch.Tensor]): Dictionary of combinations.

        Returns:
            torch.Tensor: Kalman gain.
        """
        pass

    def predict(self, fx_kwargs: dict[str, torch.Tensor] = {}) -> None:
        """Predicts the state vector at the next time step.

        Args:
            fx (callable): State transition function.
            fx_kwargs (dict[str, torch.Tensor]): Additional arguments for the state transition function.

        Returns:
            None
        """
        # Call the state transition function to get the predicted state vector
        # Do a state transition along the batch dimension
        self.x_prior = torch.vstack(
            [self.fx(self.x_posterior[i], **fx_kwargs) for i in range(self.batch_dim)]
        )

    def _get_f1(self) -> torch.Tensor:
        """Get the F1 combination.

        Returns:
            torch.Tensor: The F1 combination. Shape (ba
        """
        if not self.measuremnt_tracker.is_full():
            return torch.zeros(
                self.batch_dim,
                self.dim_measurement,
                device=self.device,
                dtype=self.dtype,
            )

        return self.F1(
            y_curr=self.measuremnt_tracker[-1], y_prev=self.measuremnt_tracker[-2]
        )

    def _get_f4(self) -> torch.Tensor:
        """Get the F4 combination.

        Returns:
            torch.Tensor: The F4 combination.
        """
        if not self.state_tracker.is_full():
            return torch.zeros(
                self.batch_dim, self.dim_state, device=self.device, dtype=self.dtype
            )

        return self.F4(
            x_prior=self.state_tracker[-1], x_posterior=self.state_tracker[-2]
        )

    def update(self, y: torch.Tensor, hx_kwargs: dict[str, torch.Tensor] = {}) -> None:
        """Updates the state vector based on the measurement.

        Args:
            y (torch.Tensor): Measurement vector. Shape (batch_size, dim_measurement).
            hx_kwargs (dict[str, torch.Tensor]): Additional arguments for the measurement function.

        Returns:
            None
        """
        # Check if the measurement has the proper shape
        if y.shape != (self.batch_dim, self.dim_measurement):
            raise ValueError(
                f"The measurement vector should have the shape {(self.batch_dim, self.dim_measurement)}"
            )

        # Add the measurement to the measurement tracker
        self.measuremnt_tracker.add(y)

        # Call the measurement function to get the predicted measurement
        # Do a measurement along the batch dimension
        y_pred = torch.vstack(
            [self.hx(self.x_prior[i], **hx_kwargs) for i in range(self.batch_dim)]
        )

        # Calculate the combinations
        combinations = {
            "F1": self._get_f1(),  # Safely calls the F1 combination even if the tracker is not full
            "F2": self.F2(y, y_pred),
            "F3": self.F3(self.x_prior, self.x_posterior),
            "F4": self._get_f4(),  # Safely calls the F4 combination even if the tracker is not full
        }

        # Calculate the Kalman gain
        K = self.calculate_kalman_gain(combinations)

        # Update the state vector along the batch dimension
        x = torch.vstack(
            [
                self.x_prior[i] + torch.matmul(K[i], y[i] - y_pred[i])
                for i in range(self.batch_dim)
            ]
        )

        # Add the state vector to the state tracker
        self.state_tracker.add(
            x.detach()
        )  # Make sure to detach the state vector to avoid backpropagation since this is just data now not a parameter

        # Set the posterior state vector
        self.x_posterior = (
            x.detach()
        )  # Make sure to detach the state vector to avoid backpropagation since this is just data now not a parameter

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This methods is convenience method to process the batch of data at once instead of calling standerd predict and update methods.

        This method is useful for training the filter in a batch way. The kalman net are by definition used for
        time series data. This method is useful for training the filter for multiple trajectories at once. The
        input to this method should be the trajectory data of shape (trajecectory_length, batch_size, dim_state)
        where trajectory_length is the number of timestamps in the trajectory, batch_size is the number of time series
        being tracked and dim_state is the dimension of the state vector at each time step.

        For example, if one is tracking the position and velocity of a single object, the state vector will be of (x, vx)
        but if one wants to train it using batch data the state vector will be of (batch_size, state_dim) where batch_size.
        However, one wants to train the filter in multiple trajectories at once, i.e from jan 1 to jan 10,  and jan 10 to jan 20.
        User can stack the trajectories and pass it to this method.

        Args:
            x (torch.Tensor): Current state vector. Shape

        Returns:
            torch.Tensor: Predicted state vector. Shape (batch_size, dim_state).
        """
        # Reset the internal state of the filter
        self.reset_internal_state(
            batch_dim=x.shape[1]
        )  # Set the batch dim to the number of batch data

        # Get the number of time steps
        num_time_steps = x.shape[0]

        # output
        output = torch.zeros(x.shape, device=self.device, dtype=self.dtype)

        # Iterate over the time steps
        for i in range(num_time_steps):
            # Predict the state vector
            self.predict()
            output[i] = self.update(x[i])

        # Retset the internal state of the filter
        # This ensures that the filter is ready for prediction and update for the next batch of data
        self.reset_internal_state(batch_dim=1)
        return output


# Path: src/navigator/filters/kalman_nets/kalman_net.py
