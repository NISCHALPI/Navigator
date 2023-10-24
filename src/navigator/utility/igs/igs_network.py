"""This module provides the IGSNetwork class which contains the information about the IGS network, including IGS stations, network name, network ID, network description, and station details.

Attributes:
    _igs_file_path (str): A class-level attribute pointing to the path of the IGS network data file.

Classes:
    IGSNetwork: A class for managing and accessing IGS network information.

"""

import os

import numpy as np
import pandas as pd

__all__ = ["IGSNetwork"]


class IGSNetwork:
    """This class contains the information about the IGS network, including the IGS stations, network name, network ID, network description, and station details.

    Attributes:
        _igs_file_path (str): A class-level attribute pointing to the path of the IGS network data file.

    Methods:
        __init__(): Constructor method for initializing the IGSNetwork object and loading the network data from a CSV file.
        get_xyz(station: str) -> tuple: Retrieve the XYZ coordinates of a specific station in the IGS network.
        get_ellipsoid(station: str) -> str: Retrieve the ellipsoid details (Latitude, Longitude, Height) for a specific station.
        get_igs_station(station: str) -> pd.Series: Retrieve detailed information for a specific IGS station.
        __len__() -> int: Get the total number of stations in the IGS network.
        __getitem__(station: str) -> pd.Series: Retrieve detailed information for a specific IGS station using indexing.
        __contains__(station: str) -> bool: Check if a station is part of the IGS network.
        __repr__() -> str: Return a string representation of the IGSNetwork object.
        __str__() -> str: Return a string representation of the IGSNetwork object for display.
        __iter__() -> iter: Allow iteration through the IGS station names.

    Properties:
        stations (list): Read-only property to access the IGS network data as a DataFrame.

    """

    _igs_file_path = os.path.join(os.path.dirname(__file__), "IGSNetwork.csv")

    def __init__(self) -> None:
        """Constructor method for initializing the IGSNetwork object and loading the network data from a CSV file."""
        # Load the IGS network file
        self._igs_network = pd.read_csv(
            self._igs_file_path, index_col=0
        ).infer_objects()

    def get_xyz(self, station: str) -> tuple:
        """Retrieve the XYZ coordinates of a specific station in the IGS network."""
        # Check if the station is in the IGS network
        if station not in self._igs_network.index:
            raise ValueError(f"The station {station} is not in the IGS network.")

        return self._igs_network.loc[station, ["X", "Y", "Z"]].values

    def get_ellipsoid(self, station: str) -> str:
        """Retrieve the ellipsoid details (Latitude, Longitude, Height) for a specific station."""
        # Check if the station is in the IGS network
        if station not in self._igs_network.index:
            raise ValueError(f"The station {station} is not in the IGS network.")

        return self._igs_network.loc[
            station, ["Latitude", "Longitude", "Height"]
        ].values

    def get_igs_station(self, station: str) -> pd.Series:
        """Retrieve detailed information for a specific IGS station."""
        return self._igs_network.loc[station]

    def error(self, station: str, x_hat: float, y_hat: float, z_hat: float) -> float:
        """Calculate the error of a measurement with respect to a station.

        Args:
            station (str): The name of the station in the IGS network.
            x_hat (float): The estimated X coordinate of the measurement.
            y_hat (float): The estimated Y coordinate of the measurement.
            z_hat (float): The estimated Z coordinate of the measurement.

        Returns:
            float: The error of the measurement with respect to the station.
        """
        # Calculate the error of the measurement with respect to the station
        station_coord = self.get_xyz(station)

        # Calculate the error
        return np.linalg.norm(station_coord - np.array([x_hat, y_hat, z_hat]))

    @property
    def stations(self) -> list:
        """Read-only property to access the IGS network data as a DataFrame."""
        return self._igs_network

    def __len__(self) -> int:
        """Get the total number of stations in the IGS network."""
        return len(self._igs_network)

    def __getitem__(self, station: str) -> pd.Series:
        """Retrieve detailed information for a specific IGS station using indexing."""
        return self.get_igs_station(station)

    def __contains__(self, station: str) -> bool:
        """Check if a station is part of the IGS network."""
        return station in self._igs_network.index

    def __repr__(self) -> str:
        """Return a string representation of the IGSNetwork object."""
        return f"IGSNetwork(stations={len(self)})"

    def __str__(self) -> str:
        """Return a string representation of the IGSNetwork object for display."""
        return f"IGSNetwork(stations={len(self)})"

    def __iter__(self) -> iter:
        """Allow iteration through the IGS station names."""
        return iter(self._igs_network.index)

    def __call__(self, station: str, x: float, y: float, z: float) -> float:
        """Calculate the error of a measurement with respect to a station.

        Args:
            station (str): The name of the station in the IGS network.
            x (float): The estimated X coordinate of the measurement.
            y (float): The estimated Y coordinate of the measurement.
            z (float): The estimated Z coordinate of the measurement.

        Returns:
            float: The error of the measurement with respect to the station.
        """
        return self.error(station, x, y, z)
