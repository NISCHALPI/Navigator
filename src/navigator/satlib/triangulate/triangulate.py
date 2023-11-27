"""Triangulation Library for Satellite Navigation Module.

This module provides classes and functionality for performing triangulation computations based on satellite navigation data.

Classes:
    AbstractTriangulate (ABC): An abstract base class for implementing triangulation algorithms.

Attributes:
    __all__ (list): A list of names to be exported when using `from module import *`.

Methods:
    __init__(self, interface: Itriangulate, dispatcher: AbstractDispatcher = None) -> None:
        Initializes an instance of the AbstractTriangulate class with a triangulation interface and an optional dispatcher.

    _compute(self, obs: pd.DataFrame, nav: pd.DataFrame, *args, **kwargs) -> pd.Series | pd.DataFrame:
        Abstract method for computing triangulated locations. Subclasses must implement this method.

    __call__(self, obs: pd.DataFrame, nav: pd.DataFrame, *args, **kwargs) -> pd.Series | pd.DataFrame:
        Callable method for performing triangulation. Delegates to the _compute method.

    __repr__(self) -> str:
        Returns a string representation of the AbstractTriangulate instance.

Properties:
    iTriangulate (property): Property to access the triangulation interface.

    iTriangulate (setter): Setter for the triangulation interface property.

Note:
    This module is part of a triangulation library and includes an abstract base class (AbstractTriangulate) for implementing specific triangulation algorithms. Subclasses of AbstractTriangulate must implement the _compute method to provide triangulation functionality.
"""

import webbrowser
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ...dispatch.base_dispatch import AbstractDispatcher
from ...utility import Epoch
from ...utility.igs_network import IGSNetwork
from .itriangulate.itriangulate import Itriangulate

__all__ = ["AbstractTriangulate", "Triangulate"]


class AbstractTriangulate(ABC):
    """Abstract base class for triangulation algorithms.

    This class defines the interface for implementing triangulation algorithms for satellite navigation. Subclasses are expected to implement the _compute method for actual triangulation computations.

    Attributes:
        itraingulate (property): Property to access the triangulation interface.
    """

    def __init__(
        self, interface: Itriangulate, dispatcher: AbstractDispatcher = None
    ) -> None:
        """Initialize an instance of the AbstractTraingulate class.

        Args:
            interface (Itriangulate): An instance of an Itriangulate implementation.
            dispatcher (AbstractDispatcher, optional): An optional dispatcher for handling triangulation tasks. Defaults to None.

        Raises:
            TypeError: If the interface is not an instance of Itriangulate.
            TypeError: If the dispatcher is not an instance of AbstractDispatcher.
        """
        self.itriangulate = interface

        if dispatcher is not None and not isinstance(dispatcher, AbstractDispatcher):
            raise TypeError("dispatcher must be an instance of AbstractDispatcher")
        self.dispatcher = dispatcher

    @abstractmethod
    def _compute(
        self,
        obs: Epoch,
        obs_metadata: pd.Series,
        nav_metadata: pd.Series,
        *args,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Abstract method for computing triangulated locations.

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.
            obs_metadata (pd.Series): Metadata for the observation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.


        Returns:
            pd.Series | pd.DataFrame: The computed triangulated location.
        """
        return self.itriangulate._compute(
            obs, obs_metadata, nav_metadata, *args, **kwargs
        )

    def __call__(
        self,
        obs: Epoch,
        obs_metadata: pd.Series,
        nav_metadata: pd.Series,
        *args,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Abstract method for computing triangulated locations.

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.
            obs_metadata (pd.Series): Metadata for the observation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.


        Returns:
            pd.Series | pd.DataFrame: The computed triangulated location.
        """
        return self._compute(obs, obs_metadata, nav_metadata, *args, **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the AbstractTraingulate instance.

        Returns:
            str: A string representation of the instance.
        """
        return f"{self.__class__.__name__}(itraingulate={self.itriangulate}, dispatcher={self.dispatcher})"

    @property
    def itriangulate(self) -> Itriangulate:
        """Property to access the triangulation interface.

        Returns:
            Itriangulate: The triangulation interface instance.
        """
        return self._interface

    @itriangulate.setter
    def itriangulate(self, interface: Itriangulate) -> None:
        """Setter for the triangulation interface property.

        Args:
            interface (Itriangulate): An instance of Itriangulate.

        Raises:
            TypeError: If the provided interface is not an instance of Itriangulate.
        """
        if not isinstance(interface, Itriangulate):
            raise TypeError("itraingulate must be an instance of Itriangulate")

        self._interface = interface


class Triangulate(AbstractTriangulate):
    """Concrete class for triangulation algorithms.

    This class is a concrete implementation of the AbstractTriangulate class, providing specific functionality for a triangulation algorithm.

    Methods:
        _compute(self, obs: pd.DataFrame, nav: pd.DataFrame, *args, **kwargs) -> pd.Series | pd.DataFrame:
            Computes triangulated locations using a specific algorithm.

    Args:
            obs (pd.DataFrame): DataFrame containing observation data.
            nav (pd.DataFrame): DataFrame containing navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

    Returns:
            pd.Series | pd.DataFrame: The computed triangulated location.

    Summary:
        The Triangulate class is a concrete implementation of the AbstractTriangulate class, providing the _compute method to perform triangulation using a specific algorithm.
    """

    def _compute(
        self,
        obs: Epoch,
        obs_metadata: pd.Series,
        nav_metadata: pd.Series,
        *args,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Computes triangulated locations using a specific algorithm."""
        return super()._compute(obs, obs_metadata, nav_metadata, *args, **kwargs)

    def igs_diff(
        self,
        obs: Epoch,
        obs_metadata: pd.Series = None,
        nav_metadata: pd.Series = None,
        *args,
        **kwargs,
    ) -> float:
        """Computes the error between the computed location and the actual location for only IGS stations.

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.
            obs_metadata (pd.Series): Metadata for the observation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The error between the computed location and the actual location.

        Note:
            This method is only for IGS stations.
        """
        # Get the actual location from the IGS network
        actual = self.igs_real_coords(obs)

        # Get the computed location
        computed = self._compute(obs, obs_metadata, nav_metadata, *args, **kwargs)

        # Calculate the difference between the computed and actual locations
        return np.linalg.norm(computed[['x', 'y', 'z']] - actual)

    def igs_real_coords(
        self,
        obs: Epoch,
    ) -> np.ndarray:
        """Computes the actual location for only IGS stations.

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.

        Returns:
            np.ndarray: The actual location.
        """
        # Get the actual location from the IGS network
        return IGSNetwork().get_xyz(obs.station)

    def coords(
        self,
        obs: Epoch,
        obs_metadata: pd.Series = None,
        nav_metadata: pd.Series = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Computes the triangulated location.

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.
            obs_metadata (pd.Series): Metadata for the observation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The computed triangulated location.
        """
        return self._compute(obs, obs_metadata, nav_metadata, *args, **kwargs)[
            ['x', 'y', 'z']
        ].to_numpy()

    @staticmethod
    def google_earth_view(lat: float, lon: float) -> None:
        """Open Google Earth in a web browser centered at the specified coordinates.

        Args:
            lat (float): The latitude of the location.
            lon (float): The longitude of the location.

        Returns:
            None
        """
        try:
            # Construct the Google Earth URL with the specified coordinates
            url = f"https://earth.google.com/web/search/{lat},{lon}"

            # Open the URL in the default web browser
            webbrowser.open(url)

            print("Google Earth opened successfully.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        return

    @staticmethod
    def google_maps_view(lat: float, lon: float) -> None:
        """Open Google Maps in a web browser centered at the specified coordinates.

        Args:
            lat (float): The latitude of the location.
            lon (float): The longitude of the location.

        Returns:
            None
        """
        try:
            # Construct the Google Maps URL with the specified coordinates
            url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"

            # Open the URL in the default web browser
            webbrowser.open(url)

            print("Google Maps opened successfully.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        return

    @staticmethod
    def google_earth_view_cartisian(x: float, y: float, z: float) -> None:
        """Open Google Earth in a web browser centered at the specified coordinates with a marker.

        Args:
            x (float): The x coordinate of the location.
            y (float): The y coordinate of the location.
            z (float): The z coordinate of the location.

        Returns:
            None
        """
        try:
            # Construct the Google Earth URL with the specified coordinates and a marker
            url = (
                f"https://earth.google.com/web/search/?q={x},{y},{z}&place={x},{y},{z}"
            )

            # Open the URL in the default web browser
            webbrowser.open(url)

            print("Google Earth opened successfully.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        return
