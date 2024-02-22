"""This module contains the triangulation interface for the Kalman filter for GPS/GNSS triangulation.

Classes:
    KalmanTriangulationInterface

This interface is generic and can be used with any state definition and measurement models. All the Kalman filters implemented in this package should inherit from this interface.

"""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from pandas.core.api import DataFrame, Series

from .....epoch.epoch import Epoch
from .....utility.transforms.coordinate_transforms import geocentric_to_ellipsoidal
from ....satellite.iephm.sv.tools.elevation_and_azimuthal import elevation_and_azimuthal
from ..iterative.iterative_traingulation_interface import (
    IterativeTriangulationInterface,
)
from ..itriangulate import Itriangulate


class KalmanTriangulationInterface(Itriangulate, ABC):
    """This class provides the interface for the Kalman filter for GPS/GNSS triangulation."""

    def __init__(self, num_sv: int, dt: float, interface_id: str) -> None:
        """Initializes an instance of the Kalman Triangulation Interface.

        Args:
            num_sv (int): The number of satellites to track.
            dt (float): The sampling time interval in seconds.
            interface_id (str): The identifier for the interface like 'EKF', 'UKF', 'PF' etc.
        """
        if num_sv < 2:
            raise ValueError(
                "The number of satellites to track must be greater than 2."
            )
        self.num_sv = num_sv

        if dt <= 0:
            raise ValueError(
                f"The sampling time interval must be positive but got {dt}."
            )
        self.dt = dt

        super().__init__(feature=interface_id)

    @staticmethod
    def _clipped_geocentric_to_ellipsoidal(x: float, y: float, z: float) -> tuple:
        """Convert the geocentric coordinates to ellipsoidal coordinates.

        Args:
            x (float): The x-coordinate in the geocentric frame.
            y (float): The y-coordinate in the geocentric frame.
            z (float): The z-coordinate in the geocentric frame.

        Returns:
            tuple: A tuple containing the ellipsoidal coordinates (latitude, longitude, height).
        """
        return geocentric_to_ellipsoidal(
            np.clip(x, -740000, 740000),
            np.clip(y, -740000, 740000),
            np.clip(z, -740000, 740000),
        )

    def _trim_by_first_n(
        self, pseudorange: np.ndarray, sv_coords: DataFrame
    ) -> tuple[np.ndarray, DataFrame]:
        """Trim the pseudorange and sv_coords to the first n satellites.

        Args:
            pseudorange (np.ndarray): The range measurements.
            sv_coords (DataFrame): The coordinates of the satellites indexed by PRN.

        Returns:
            tuple[np.ndarray, DataFrame]: A tuple containing the trimmed range and sv_coords.
        """
        # Raise an error if the number of satellites is greater than the number of satellites in the epoch
        if self.num_sv > len(pseudorange):
            raise ValueError(
                f"The number of satellites  to track  {self.num_sv} cannot be greater than the number of satellites in the epoch {len(pseudorange)}."
            )
        # Trim the epoch to the required number of satellites
        pseudorange = pseudorange[: self.num_sv]
        sv_coords = sv_coords.iloc[: self.num_sv]
        return pseudorange, sv_coords

    def _trim_by_elevation(
        self,
        pseudorange: np.ndarray,
        sv_coords: DataFrame,
        observer_position: np.ndarray,
    ) -> tuple[Series, DataFrame]:
        """Trim the range and sv_coords to the first n satellites with the highest elevation.

        Args:
            pseudorange (np.ndarray): The range measurements.
            sv_coords (DataFrame): The coordinates of the satellites indexed by PRN.
            observer_position (np.ndarray): The observer position  from which the elevation is calculated.

        Returns:
            tuple[pd.Series, pd.DataFrame]: A tuple containing the trimmed range and sv_coords.
        """
        # Raise an error if the number of satellites is greater than the number of satellites in the epoch
        if self.num_sv > len(pseudorange):
            raise ValueError(
                f"The number of satellites  to track  {self.num_sv} cannot be greater than the number of satellites in the epoch {len(pseudorange)}."
            )

        # Attach the range to the sv_coords
        # Just easier to work rather than using the index
        sv_coords["range"] = pseudorange

        # If elevation is not available,  calculate it
        # Sometimes the elevation is already available
        if "elevation" not in sv_coords.columns:
            # Get the elevation and azimuthal angles
            elevation, _ = elevation_and_azimuthal(
                satellite_positions=sv_coords[["x", "y", "z"]].values,
                observer_position=observer_position,
            )

            # Sort the  prns by elevation
            sv_coords["elevation"] = elevation

        sv_coords = sv_coords.sort_values(by="elevation", ascending=False)

        # Grab the first n satellites having the highest elevation
        sv_coords = sv_coords.iloc[: self.num_sv]

        # Drop the elevation column
        sv_coords = sv_coords.drop(columns=["elevation"])

        return sv_coords["range"], sv_coords

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
            - lat: The latitude of the position.
            - lon: The longitude of the position.
            - height: The height of the position.
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

    def _get_least_squared_estimation(self, epoch: Epoch) -> np.ndarray:
        """Get the least squared estimation of the position,  clock bias, and covariance matrix.

        Args:
            epoch (Epoch): The epoch to be processed.

        TODO: This method only works for GPS. It should be generalized to work for any GNSS.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the least squared estimation of the position and the covariance matrix.  [4,4]
        """
        # Set the epoch profile to init
        epoch.profile = Epoch.INITIAL

        # Get the range and sv_coordinates
        (
            coords,
            Q,
            sigma,
        ) = IterativeTriangulationInterface()._get_coords_and_covar_matrix(
            epoch=deepcopy(epoch),
        )
        coords = coords.flatten()
        return coords, Q
