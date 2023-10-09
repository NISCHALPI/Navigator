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
    # Create an 'Iephemeris' object
    ephemeris = SomeEphemerisImplementation()

    # Create a 'Satellite' object
    satellite = Satellite(iephemeris=ephemeris)

    # Compute satellite positions
    positions = satellite(metadata=metadata, data=data)

"""


from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import numpy as np

from ..iephm import AbstractIephemeris

__all__ = ['AbstractSatellite', 'Satellite']


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
        """_summary_.

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
        Tsv: pd.DataFrame | pd.Timestamp | str,
        metadata: pd.Series,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Abstract method to compute satellite position based on ephemeris data.

        Args:
            Tsv (pd.DataFrame[pd.Timestamp] | pd.Timestamp): Timestamps for which to compute satellite positions announced by respective satellites.
                If a DataFrame is passed, it must have same index as the data and a column name Tsv for sat specifice time stamps.
            metadata (pd.Series): Metadata related to the ephemeris data.
            data (pd.DataFrame): Data from the navigation file.

        Returns:
            pd.DataFrame: A DataFrame representing the calculated satellite positions.
        """
        # Change to time stamp if string is passed
        if isinstance(Tsv, str):
            Tsv = pd.Timestamp(Tsv)

        # Instantiate empty dataframe to store satellite position
        position_df = pd.DataFrame(index=data.index, columns=['x', 'y', 'z'])

        for (time, sv), rowdata in data.iterrows():
            # Attach time and sv to rowdata
            rowdata['Toc'] = time
            rowdata['sv'] = sv

            # Get the t_sv for respective satellite
            if isinstance(Tsv, pd.DataFrame):
                rowdata['Tsv'] = Tsv.loc[(time, sv), 'Tsv']
            elif isinstance(Tsv, (pd.Timestamp, datetime)):
                rowdata['Tsv'] = Tsv
            else:
                raise TypeError("t_sv must be a DataFrame or Timestamp")

            # Compute satellite position using ephemeris data and iepehemeris interface
            position_df.loc[(time, sv), ['x', 'y', 'z']] = self._iephemeris(
                metadata, rowdata
            )

        return position_df

    def __call__(
        self,
        t_sv: pd.DataFrame | pd.Timestamp | str,
        metadata: pd.Series,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Computes satellite position based on ephemeris data.

        Args:
            t_sv (pd.DataFrame[pd.Timestamp] | pd.Timestamp): Timestamps for which to compute satellite positions announced by respective satellites.
                If a DataFrame is passed, it must have same index as the data and a column name t_sv for sat specifice time stamps.
            metadata (pd.Series): Metadata related to the ephemeris data.
            data (pd.DataFrame): Data related to the ephemeris data.

        Returns:
            pd.DataFrame: A DataFrame representing the calculated satellite positions.
        """
        return self._compute(t_sv, metadata, data)

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
        t_sv: pd.DataFrame | pd.Timestamp | str,
        metadata: pd.Series,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        return super()._compute(t_sv, metadata, data)

    def trajectory(
        self,
        t_sv: str | pd.Timestamp,
        metadata: pd.Series,
        data: pd.DataFrame,
        interval: int = 3600,
        step: int = 10,
    ) -> np.ndarray:
        """Computes satellite trajectory based on ephemeris data. The trajectory is computed and returned as a numpy array in the following format, [SV, 3, step] where SV is original satellite index,
        Coordinates are x,y,z coordinates in ECEF frame and step are the number of points in the trajectory calculated by using interval/step. The final shape of the array is (SV, 3, step).

        Args:
            t_sv (pd.DataFrame | pd.Timestamp | str): Timestamps for which to start satellite positions interpolation.
            metadata (pd.Series): Metadata related to the ephemeris data.
            data (pd.DataFrame): RINEX navigation data and Ephemeris data.
            interval (int, optional): The time in seconds from t_sv to compute satellite trajectory. Defaults to 3600 seconds.
            step (int, optional): The number of time divisions in the interval .i.e every n seconds the satellite position is computed. Defaults to every 10 seconds.

        Returns:
            np.ndarray: A numpy array representing the calculated satellite trajectory in format [SV, 3 , step].
        """
        # Convert to timestamp
        if isinstance(t_sv, str):
            t_sv = pd.Timestamp(t_sv)

        # TO DO : Currently only works for GPS satellites
        # Check if the interface is GPS
        if not self._iephemeris._feature == 'GPS':
            raise NotImplementedError(
                "Currently only GPS interface is supported for trajectory calculation"
            )
        # Satellite position stack
        sat_pos_list = []

        # Loop over interval and compute satellite position
        for _ in range(0, interval, step):
            sat_pos_list.append(self(t_sv, metadata, data))
            t_sv += pd.Timedelta(seconds=step)

        # Convert to array
        sat_pos_list = [frame.to_numpy() for frame in sat_pos_list]

        # Return the stacked array
        return np.stack(sat_pos_list, axis=-1)
