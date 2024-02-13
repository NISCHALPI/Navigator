"""This module contains the interface for all Kalman Filter based triangulation methods.

The interface is designed to be used as a base class for all Kalman Filter based triangulation methods.

Classes:
    - `KalmanTriangulationInterface`: Interface for all Kalman Filter based triangulation methods.

"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from navigator.epoch.epoch import Epoch

from .....utility.transforms.coordinate_transforms import geocentric_to_ellipsoidal
from ..itriangulate import Itriangulate


class KalmanTriangulationInterface(Itriangulate, ABC):
    """Interface for all Kalman Filter based triangulation methods.

    This class is designed to be used as a base class for all Kalman Filter based triangulation methods.

    """

    def __init__(self, dt: float, num_satellite: int, saver: bool = False) -> None:
        """Initializes an instance of the KalmanTriangulationInterface.

        Args:
            dt (float): The sampling time interval in seconds.
            num_satellite (int): Number of satellites to track using the Kalman Filter.
            saver (bool, optional): Whether to save the intermediate results. Defaults to False.
        """
        # Check if the number of satellites to track is valid
        if num_satellite < 4:
            raise ValueError(
                "Number of satellites to track must be greater than or equal to 4."
            )
        self.num_satellite = num_satellite
        self.saver = saver

        # Check if the sampling time interval is valid
        if dt <= 0:
            raise ValueError("Sampling time interval must be greater than 0.")
        self.dt = dt

    @abstractmethod
    def epoch_profile(self) -> dict:
        """Set the profile of the epoch for the preprocessor.

        Epoch profile is dependens on what corrections are to be applied to the data. For eg. Tropospheric correction, Ionospheric correction etc
        before the data is fed to the Kalman Filter.

        Returns:
            dict: The profile of the preprocessor on how it will preprocess the data. For eg {"apply_tropo": True, "apply_iono": False}

        """
        pass

    def output(self, coords: np.ndarray) -> pd.Series:
        """Output the estimated position and clock bias.

        This method is here to ensure that the output of the Kalman Filter based triangulation method is consistent in
        naming convention with the output of the other triangulation methods. At least these outputs must be estimated by the Kalman Filter.
        Other estimates can be added by the inheriting class.

        Args:
            coords (np.ndarray): The estimated position coordinates.

        Returns:
            pd.Series: The estimated position in ECEF coordinates, geodetic coordinates and clock bias.
        """
        # Convert the ECEF coordinates to geodetic coordinates
        # Clip for numerical stability
        x_clip, y_clip, z_clip = (
            np.clip(coords, -740000, 740000),
            np.clip(coords, -740000, 740000),
            np.clip(coords, -740000, 740000),
        )

        # Convert the ECEF coordinates to geodetic coordinates
        lat, lon, h = geocentric_to_ellipsoidal(x_clip, y_clip, z_clip)

        return pd.Series(
            {
                "x": coords[0],
                "y": coords[1],
                "z": coords[2],
                "lat": lat,
                "lon": lon,
                "height": h,
            }
        )


class FilterPyBasedKalmanInterface(KalmanTriangulationInterface):
    """Interface for all Kalman Filter based triangulation methods.

    This class is designed to be used as a base class for all Kalman Filter based triangulation methods.

    """

    def __init__(self, dt: float, num_satellite: int, saver: bool = False) -> None:
        """Initializes an instance of the KalmanTriangulationInterface.

        Args:
            dt (float): The sampling time interval in seconds.
            num_satellite (int): Number of satellites to track using the Kalman Filter.
            saver (bool, optional): Whether to save the intermediate results. Defaults to False.
        """
        # Check if the number of satellites to track is valid
        if num_satellite < 4:
            raise ValueError(
                "Number of satellites to track must be greater than or equal to 4."
            )
        self.num_satellite = num_satellite
        self.saver = saver
        # Check if the sampling time interval is valid
        if dt <= 0:
            raise ValueError("Sampling time interval must be greater than 0.")
        self.dt = dt

    def output_hook(self) -> dict:
        """Output hook for the inheriting class to add additional outputs.

        Returns:
            dict: Additional outputs estimated by the inheriting class.
        """
        return {}

    def _compute(self, epoch: Epoch, *args, **kwargs) -> pd.Series | pd.DataFrame:
        """Computes the triangulated position using the Kalman Filter.

        Args:
            epoch (Epoch):  The epoch to be processed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.


        Returns:
            pd.Series | pd.DataFrame: The estimated position in ECEF coordinates, geodetic coordinates and clock bias.
        """
        # Check if the lenth of epoch is valid
        if len(epoch) < self.num_satellite:
            raise ValueError(
                f"Epoch must have at least {self.num_satellite} satellites."
            )
        # Remove the extra satellites from the epoch
        epoch.obs_data = epoch.obs_data.iloc[: self.num_satellite]
        epoch.trim()  # Remove the extra satellites from the navigation data

        # Get the range and sv_coordinates
        # Update the profile of epoch
        epoch.profile.update(self.epoch_profile())

        # Preprocess the data
        pseudorange, sv_coordinates = self._preprocess(
            epoch=epoch,
            **kwargs,
        )

        # Run the Kalman Filter Pred and Update Loop
        self.predict()  # Prediction step does not require any arguments for GNSS data. # TODO: Add the arguments for the prediction step if required
        # Update the state vector using the Kalman Filter, requires the pseudorange and the satellite coordinates at least
        self.update(
            pseudorange.values,
            sv_location=sv_coordinates[["x", "y", "z"]].values,
            **kwargs,
        )

        # Save the intermediate results if saver is specified
        if self.saver:
            self.save()

        # Get the current state vector
        results = self.output(self.get_current_coordinat())

        # Update output hook
        results.update(self.output_hook())

        return results

    @abstractmethod
    def predict(self, *args, **kwargs) -> None:
        """Predict the next state vector using the Kalman Filter.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the state vector using the Kalman Filter.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        pass

    @abstractmethod
    def save(self) -> None:
        """Implements how to save the intermediate results of the Kalman Filter."""
        pass

    @abstractmethod
    def get_current_coordinat(self) -> np.ndarray:
        """Get the current coordinates vector estimated by the Kalman Filter.

        Returns:
            np.ndarray: The current state vector.
        """
        pass
