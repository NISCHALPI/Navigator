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
from copy import deepcopy

import numpy as np
import pandas as pd
import tqdm

from ...dispatch.base_dispatch import AbstractDispatcher
from ...epoch import Epoch
from ...utility.transforms.coordinate_transforms import geocentric_to_enu
from .itriangulate.iterative.iterative_traingulation_interface import (
    IterativeTriangulationInterface,
)
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
        epoch: Epoch,
        *args,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Abstract method for computing triangulated locations.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.


        Returns:
            pd.Series | pd.DataFrame: The computed triangulated location.
        """
        return self.itriangulate._compute(
            deepcopy(epoch),  # Deep copy the epoch to avoid modifying the original
            *args,
            **kwargs,
        )

    def __call__(
        self,
        epoch: Epoch,
        *args,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Abstract method for computing triangulated locations.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.


        Returns:
            pd.Series | pd.DataFrame: The computed triangulated location.
        """
        return self._compute(epoch, *args, **kwargs)

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

    @staticmethod
    def _get_initial_approx_using_wls(
        epoch: Epoch,
        **kwargs,
    ) -> pd.Series:
        """Computes the initial approximation for the reciever position using a weighted least squares method.

        This is needed for most triangulation algorithms to provide an initial approximation for the receiver position to apply the
        approximate location dependedn error model. This method computes the initial approximation using a weighted least squares method
        that is crude since no error model is applied.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: The initial approximation.
        """
        # Copy  the epoch to avoid modifying the original
        epoch_inital = deepcopy(epoch)
        # Set the initial  epoch profile

        # Initial profile doesn;t need any prior approximation of user position since no error model is applied
        # Check that it is not the dummy profile
        if epoch_inital.profile["mode"] != "dummy":
            epoch_inital.profile = Epoch.INITIAL
        # Compute the initial approximation using the weighted least squares method
        return IterativeTriangulationInterface()._compute(epoch=epoch_inital, **kwargs)


class Triangulate(AbstractTriangulate):
    """Concrete class for triangulation algorithms.

    This class is a concrete implementation of the AbstractTriangulate class, providing specific functionality for a triangulation algorithm.

    Attributes:
        itraingulate (property): Property to access the triangulation interface.


    Methods:
        _compute(self, epoch: Epoch, *args, **kwargs) -> pd.Series | pd.DataFrame:
            Computes triangulated locations using a specific algorithm.

        igs_diff(self, epoch: Epoch, *args, **kwargs) -> pd.Series:
            Computes the error between the computed location and the actual location for only IGS stations.

        igs_real_coords(self, obs: Epoch) -> np.ndarray:
            Computes the actual location for only IGS stations.

        coords(self, epoch: Epoch, *args, **kwargs) -> np.ndarray:
            Computes the triangulated location.

        triangulate_time_series(self, epochs: list[Epoch], override: bool = False, **kwargs) -> pd.DataFrame:
            Computes the triangulated location for a time series of epochs.

        google_earth_view(self, lat: float, lon: float) -> None:
            Open Google Earth in a web browser centered at the specified coordinates.

        google_maps_view(self, lat: float, lon: float) -> None:
            Open Google Maps in a web browser centered at the specified coordinates.

        google_earth_view_cartisian(self, x: float, y: float, z: float) -> None:
            Open Google Earth in a web browser centered at the specified coordinates with a marker.


    Returns:
            pd.Series | pd.DataFrame: The computed triangulated location.

    Summary:
        The Triangulate class is a concrete implementation of the AbstractTriangulate class, providing the _compute method to perform triangulation using a specific algorithm.
    """

    def _compute(
        self,
        epoch: Epoch,
        *args,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Computes triangulated locations using a specific algorithm."""
        return super()._compute(epoch, *args, **kwargs)

    def diff(
        self,
        epoch: Epoch,
        *args,
        **kwargs,
    ) -> pd.Series:
        """Computes the error between the computed location and the actual location for the epoch.

        Note: Epoch real_coords must be populated with the actual coordinates of the station.


        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            obs_metadata (pd.Series): Metadata for the observation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: The computed error with the actual location.

        Note:
            This method is only for IGS stations.
        """
        # Get the actual location from the IGS network
        if epoch.real_coord.empty:
            raise ValueError(
                "Epoch real_coords must be populated with the actual coordinates of the station."
            )

        # NOTE : Guranteed to have these coordinate in the real_coord attribute if it's not empty
        actual = epoch.real_coord[["x", "y", "z"]].to_numpy()

        # Get the computed location
        computed = self._compute(epoch, *args, **kwargs)

        # Calculate the difference between the computed and actual locations
        computed["diff"] = np.linalg.norm(computed[["x", "y", "z"]] - actual)
        return computed

    def coords(
        self,
        epoch: Epoch,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Computes the triangulated location.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The computed triangulated location.
        """
        return self._compute(epoch, *args, **kwargs)[["x", "y", "z"]].to_numpy()

    def triangulate_time_series(
        self,
        epoches: list[Epoch],
        override: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Computes the triangulated location for a time series of epochs.

        Args:
            epoches (list[Epoch]): A list of Epochs containing observation data and navigation data.
            override (bool): A flag to override errors in triangulation. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: The computed triangulated location for the time series of epochs.
        """
        # Choose the compute function based on the type of epoch
        # If the epoch contains real coordinates, use the diff method
        compute_func = None
        if all([not epoch.real_coord.empty for epoch in epoches]):
            compute_func = self.diff
        else:
            compute_func = self._compute

        # Compute the triangulated location for each epoch
        results = []

        # Get the initial approximation using the first epoch
        prior = self._get_initial_approx_using_wls(epoches[0])

        with tqdm.tqdm(total=len(epoches), desc="Triangulating") as pbar:
            for e in epoches:
                try:
                    results.append(compute_func(e, prior=prior, **kwargs))
                    prior = results[-1]
                    pbar.update(1)
                except Exception as e:
                    # If override is set, continue to the next epoch
                    if override:
                        continue
                    # Otherwise, break and raise the error
                    raise e

        return pd.DataFrame(results)

    @staticmethod
    def enu_error(predicted: pd.DataFrame, actual: pd.Series) -> pd.DataFrame:
        """Computes the error between the computed location and the actual location in ENU coordinates.

        Args:
            predicted (pd.DataFrame): The predicted location in WG84 coordinates.
            actual (pd.Series): The actual location in WG84 coordinates.

        Returns:
            pd.DataFrame: The computed error in meters. ["E_error", "N_error", "U_error"]
        """
        # Check if the actual location has x, y, z coordinates
        if all([coord not in actual.index for coord in ["x", "y", "z"]]):
            raise ValueError(
                "Actual location must have x, y, z coordinates in WG84 coordinates."
            )
        # Check if the predicted location has x, y, z coordinates
        if all([coord not in predicted.columns for coord in ["x", "y", "z"]]):
            raise ValueError(
                "Predicted location must have x, y, z coordinates in WG84 coordinates."
            )
        # Grab the unit vectors e, n, u at the actual location
        e_hat, n_hat, u_hat = geocentric_to_enu(
            x=actual["x"],
            y=actual["y"],
            z=actual["z"],
        )

        # Error vector from actual to predicted location
        error = predicted[["x", "y", "z"]] - actual[["x", "y", "z"]]

        # Project the error vector onto the unit vectors
        E_error = np.dot(error, e_hat)
        N_error = np.dot(error, n_hat)
        U_error = np.dot(error, u_hat)

        # Return the error in ENU coordinates
        return pd.DataFrame(
            {"E_error": E_error, "N_error": N_error, "U_error": U_error},
            index=predicted.index,
        )

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
