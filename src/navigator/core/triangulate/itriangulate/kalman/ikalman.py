"""This module contains the triangulation interface for the Kalman filter for GPS/GNSS triangulation.

Classes:
    IKalman - The interface for the Kalman filter for GPS/GNSS triangulation.

This interface is generic and can be used with any state definition and measurement models. All the Kalman filters implemented in this package should inherit from this interface.

"""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from pandas.core.api import DataFrame, Series

from .....epoch.epoch import Epoch
from ...itriangulate import Itriangulate
from ..iterative.iterative_traingulation_interface import (
    IterativeTriangulationInterface,
)


class IKalman(Itriangulate, ABC):
    """This class provides the interface for the Kalman filter for GPS/GNSS triangulation."""

    def __init__(self, num_sv: int, dt: float, filter_name: str) -> None:
        """Initializes an instance of the Kalman Triangulation Interface.

        Args:
            num_sv (int): The number of satellites to track.
            dt (float): The sampling time interval in seconds.
            filter (str): The filter name to be used for the Kalman filter. i.e. EKF, UKF, etc.
        """
        if num_sv < 4:
            raise ValueError(
                "The number of satellites to track must be greater than 4."
            )
        self.num_sv = num_sv

        if dt <= 0:
            raise ValueError(
                f"The sampling time interval must be positive but got {dt}."
            )
        self.dt = dt

        super().__init__(feature=filter_name)

    @abstractmethod
    def process_state(self, state: np.ndarray) -> Series:
        """Process the state vector.

        Args:
            state (np.ndarray): The state vector.

        Returns:
            Series: A pandas series containing the output provided to the user.

        Note:  This method must return at least the following keys:
            - x: The x-coordinate of the position.
            - y: The y-coordinate of the position.
            - z: The z-coordinate of the position.
        """
        pass

    @abstractmethod
    def predict_update_loop(
        self, ranges: np.ndarray, sv_coords: np.ndarray
    ) -> np.ndarray:
        """Runs the predict and update loop of the Kalman filter.

        Args:
            ranges (np.ndarray): The range measurements.
            sv_coords (np.ndarray): The coordinates of the satellites.

        Returns:
            np.ndarray: The state vector after the predict and update loop.
        """
        pass

    @abstractmethod
    def epoch_profile(self) -> str:
        """Get the epoch profile for the respective Kalman filter.

        Some might apply Ionospheric correction, tropospheric correction, etc while
        others might not. This is controlled by the epoch profile set to the epoch.

        Returns:
            str: The epoch profile that is updated for each epoch.
        """
        pass

    @staticmethod
    def autocorrelation(innovations: np.ndarray, lag: int) -> np.ndarray:
        """Compute the autocorrelation of the innovations.

        Args:
            innovations (np.ndarray): The innovations of the Kalman filter of shape (T, dim_z).
            lag (int): The lag at which to compute the autocorrelation at.

        Returns:
            np.ndarray: The autocorrelation of the innovations.
        """
        # Check if lag is valid
        if lag < 0 or lag >= innovations.shape[0]:
            raise ValueError(
                f"The lag value must be in the range [0, {innovations.shape[0]})."
            )

        # Compute the mean of the innovations
        innovations_mean = np.mean(innovations, axis=0)
        centered_innovations = innovations - innovations_mean

        # Compute the autocorrelation
        corr = 0
        for i in range(innovations.shape[0] - lag):
            corr += np.outer(centered_innovations[i], centered_innovations[i + lag])

        # Return the unbiased estimate of the autocorrelation
        return corr / (innovations.shape[0] - lag)

    @abstractmethod
    def _dim_state(self) -> int:
        """Get the dimension of the state vector.

        Returns:
            int: The dimension of the state vector.
        """
        pass

    @property
    def dim_x(self) -> int:
        """Get the dimension of the state vector.

        Returns:
            int: The dimension of the state vector.
        """
        return self._dim_state()

    @property
    def dim_z(self) -> int:
        """Get the dimension of the measurement vector.

        Returns:
            int: The dimension of the measurement vector.
        """
        return self.num_sv
