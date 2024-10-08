"""Implements the satellite class to compute satellite positions based on ephemeris data.

Based on the builder design pattern. Uses Iephemeris interface to compute satellite positions.
Will compute satellite positions based on ephemeris data and batch processing.

This module defines an abstract base class 'AbstractSatellite' that provides the blueprint for
creating satellite objects. Satellite objects are responsible for computing satellite positions
based on ephemeris data using an implementation of the 'Iephemeris' interface. The satellite
positions can be computed in batch processing mode.

Classes:
    - AbstractSatellite: Abstract base class for satellite objects.
    - Satellite: Concrete implementation of the 'AbstractSatellite' class.

Usage:
    To create a satellite object and compute satellite positions, first create an instance
    of 'AbstractSatellite' or 'Satellite', passing an 'Iephemeris' object as an argument.
    Then, call the object with metadata and data to compute satellite positions.

Example:
    >>> from navigator.core import Satellite, IGPSEphemeris
    >>> satellite_processor = Satellite(interface=IGPSEphemeris())
    >>> satellite_processor(filename=nav_dataframe)
"""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .iephm import AbstractIephemeris

__all__ = ["AbstractSatellite", "Satellite"]


class AbstractSatellite(ABC):
    """Abstract base class for satellite objects.

    This abstract class defines the blueprint for creating satellite objects that compute
    satellite positions based on ephemeris data using an implementation of the 'Iephemeris'
    interface. Satellite positions can be computed in batch processing mode.

    Args:
        iephemeris (AbstractIephemeris): An implementation of the 'AbstractIephemeris' interface
            used for satellite position calculations.

    Raises:
        TypeError: If 'iephemeris' is not an instance of 'AbstractIephemeris'.

    Methods:
        - _compute(metadata: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
          Abstract method to compute satellite position based on ephemeris data.

        - __call__(metadata: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
          Computes satellite position based on ephemeris data.

        - __repr__() -> str:
          Returns a string representation of the object.

    Usage:
        To create a concrete satellite object, inherit from this abstract class and provide
        the necessary implementation for the '_compute' method.
    """

    def __init__(
        self,
        iephemeris: AbstractIephemeris,
    ) -> None:
        """Initialize a new instance of the 'AbstractSatellite' class.

        Args:
            iephemeris (AbstractIephemeris): _description_

        Raises:
            TypeError: _description_
        """
        # Ephemeris Interface for SV location calculation

        if not issubclass(iephemeris.__class__, AbstractIephemeris):
            raise TypeError("iephemeris must be an subclass of AbstractIephemeris")

        self._iephemeris = iephemeris

        super().__init__()

    @abstractmethod
    def _compute(
        self,
        t: pd.DataFrame | pd.Timestamp | str,
        metadata: pd.Series,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Abstract method to compute satellite position based on ephemeris data.

        Args:
            t (pd.DataFrame[pd.Timestamp] | pd.Timestamp): The SV time at which to compute the satellite position.
            metadata (pd.Series): Metadata related to the ephemeris data.
            data (pd.DataFrame): Data from the navigation file.
            kwargs: Keyword arguments to be passed to the '_iephemeris' method.

        Returns:
            pd.DataFrame: A DataFrame representing the calculated satellite positions.
        """
        pass

    def __call__(
        self,
        t: pd.DataFrame | pd.Timestamp | str,
        metadata: pd.Series,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Computes satellite position based on ephemeris data.

        Args:
            t (pd.DataFrame[pd.Timestamp] | pd.Timestamp): The SV time at which to compute the satellite position.
            metadata (pd.Series): Metadata related to the ephemeris data.
            data (pd.DataFrame): Data related to the ephemeris data.
            **kwargs: Keyword arguments to be passed to the '_iephemeris' method.

        Returns:
            pd.DataFrame: A DataFrame representing the calculated satellite positions.
        """
        return self._compute(t, metadata, data, **kwargs)

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns:
            str: A string representation of the 'AbstractSatellite' object.
        """
        return f"{self.__class__.__name__}(iephemeris = {self._iephemeris!r})"


class Satellite(AbstractSatellite):
    """Concrete implementation of the 'AbstractSatellite' class.

    This class inherits from 'AbstractSatellite' and provides a concrete implementation
    for computing satellite positions based on ephemeris data.

    Methods:
        - _compute(metadata: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
          Overrides the abstract '_compute' method to provide a concrete implementation.

    Usage:
        Instantiate this class and pass an 'Iephemeris' object to compute satellite positions.
    """

    def _compute(
        self,
        t: pd.DataFrame | pd.Timestamp | str,
        metadata: pd.Series,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Computes satellite position based on ephemeris data.

        Args:
            t (pd.DataFrame[pd.Timestamp] | pd.Timestamp): The SV time at which to compute the satellite position.
            metadata (pd.Series): Metadata related to the ephemeris data.
            data (pd.DataFrame): Data from the navigation file.
            **kwargs: Keyword arguments to be passed to the '_iephemeris' method.

        Returns:
            pd.DataFrame: A DataFrame representing the calculated satellite positions.
        """
        # Change to time stamp if string is passed
        if isinstance(t, str):
            t = pd.Timestamp(t)

        # Instantiate empty dataframe to store satellite position
        position_df = []

        for (time, sv), rowdata in data.iterrows():
            # Attach time and sv to rowdata
            rowdata["Toc"] = time
            rowdata["sv"] = sv

            # Get the time of transmission
            emissionT = t if isinstance(t, pd.Timestamp) else t[(time, sv)]

            # Compute satellite position using ephemeris data and iepehemeris interface
            position_df.append(
                self._iephemeris(t=emissionT, metadata=metadata, data=rowdata, **kwargs)
            )

        return pd.DataFrame(position_df, index=data.index)

    def trajectory(
        self,
        t_sv: str | pd.Timestamp,
        metadata: pd.Series,
        data: pd.DataFrame,
        interval: int = 3600,
        step: int = 10,
        **kwargs,
    ) -> np.ndarray:
        """Computes satellite trajectory based on ephemeris data. The trajectory is computed and returned as a numpy array in the following format, [SV, 3, step] where SV is original satellite index.

        Coordinates are x,y,z coordinates in ECEF frame and step are the number of points in the trajectory calculated by using interval/step. The final shape of the array is (SV, 3, step).

        Args:
            t_sv (pd.DataFrame | pd.Timestamp | str): Timestamps for which to start satellite positions interpolation.
            metadata (pd.Series): Metadata related to the ephemeris data.
            data (pd.DataFrame): RINEX navigation data and Ephemeris data.
            interval (int, optional): The time in seconds from t_sv to compute satellite trajectory. Defaults to 3600 seconds.
            step (int, optional): The number of time divisions in the interval .i.e every n seconds the satellite position is computed. Defaults to every 10 seconds.
            **kwargs: Keyword arguments to be passed to the '_iephemeris' method.

        Returns:
            np.ndarray: A numpy array representing the calculated satellite trajectory in format [SV, 3 , step].
        """
        # Convert to timestamp
        if isinstance(t_sv, str):
            t_sv = pd.Timestamp(t_sv)

        # TO DO : Currently only works for GPS satellites
        # Check if the interface is GPS
        if not self._iephemeris._feature == "GPS":
            raise NotImplementedError(
                "Currently only GPS interface is supported for trajectory calculation"
            )
        # Satellite position stack
        sat_pos_list = []

        # Loop over interval and compute satellite position
        for _ in range(0, interval, step):
            sat_pos_list.append(
                self(t_sv, metadata, data, **kwargs).drop(columns="SVclockBias")
            )
            t_sv += pd.Timedelta(seconds=step)

        # Convert to array
        sat_pos_list = [frame.to_numpy() for frame in sat_pos_list]

        # Return the stacked array
        return np.stack(sat_pos_list, axis=-1)

    def trajectory_plot(
        self,
        t_sv: str | pd.Timestamp,
        metadata: pd.Series,
        data: pd.DataFrame,
        interval: int = 3600,
        step: int = 10,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plots the satellite trajectory based on ephemeris data.

        Args:
            t_sv (pd.DataFrame | pd.Timestamp | str): Timestamps for which to start satellite positions interpolation.
            metadata (pd.Series): Metadata related to the ephemeris data.
            data (pd.DataFrame): RINEX navigation data and Ephemeris data.
            interval (int, optional): The time in seconds from t_sv to compute satellite trajectory. Defaults to 3600 seconds.
            step (int, optional): The number of time divisions in the interval .i.e every n seconds the satellite position is computed. Defaults to every 10 seconds.
            **kwargs: Keyword arguments to be passed to the '_iephemeris' method.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the figure and axes objects.
        """
        # Get the trajectory
        trajectory = self.trajectory(t_sv, metadata, data, interval, step, **kwargs)

        # Get the figure
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_subplot(111, projection="3d")

        for i in range(trajectory.shape[0]):  # For each satellite
            ax.plot(
                trajectory[i][0], trajectory[i][1], trajectory[i][2]
            )  # Plot the trajectory

        return fig, ax
