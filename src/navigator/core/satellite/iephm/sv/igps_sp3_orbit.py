"""Implementation of the Iephemeris interface for GPS satellites using SP3 orbit files.

This module contains the implementation of the IGPSSp3 class, which serves as an
implementation of the Iephemeris interface specifically designed for GPS satellites.
It utilizes SP3 orbit files to interpolate satellite positions at given timestamps.

The SP3 orbit files provide precise information about satellite positions at
different epochs in time.

Classes:
    IGPSSp3: Implementation of the Iephemeris interface using SP3 orbit files.

Methods in IGPSSp3:
    __init__: Initializes the IGPSSp3 class.
    _nearestSubset: Retrieves the nearest subset of data around a given time.
    _compute: Computes the satellite position at a given time using interpolation.

Usage:
    To use this module, instantiate the IGPSSp3 class and call its methods to
    compute satellite positions using SP3 orbit files.
"""

from datetime import datetime

import pandas as pd
from numpy.polynomial.polynomial import Polynomial

from ..iephm import AbstractIephemeris

__all__ = ["IGPSSp3"]


class IGPSSp3(AbstractIephemeris):
    """Initialize the IGPSSp3 class.

    This class implements the Iephemeris interface for GPS satellites
    using SP3 orbit files to interpolate the satellite position.

    The SP3 orbit files contain satellite orbit data at various time epochs.
    """

    def __init__(self) -> None:
        """Constructor method for the IGPSSp3 class."""
        super().__init__(feature="SP3 Orbit")

    def _nearestSubset(
        self, data: pd.DataFrame, t: pd.Timestamp | datetime
    ) -> pd.DataFrame:
        """Get the nearest subset of data around the given time.

        Args:
            data (pd.DataFrame): SP3 orbit data.
            t (pd.Timestamp | datetime): Given time to interpolate.

        Returns:
            pd.DataFrame: 11 data points nearest to the given time,
                          sorted by their timestamps.
        """
        # Get the nearest timestamp to the given time
        nearestIdx = (
            (data.index.to_series())
            .apply(lambda x: (x - t).total_seconds())
            .abs()
            .nsmallest(11)
            .index
        )

        # Get the subset of the data
        return data.loc[nearestIdx].sort_index()

    # For compatibility with AbstractIephemeris
    def _compute(
        self,
        t: pd.Timestamp | datetime,
        metadata: pd.Series,  # noqa: ARG002
        data: pd.DataFrame,
        tolerance: int = 5,
        **kwargs,  # noqa: ARG002
    ) -> pd.Series:
        """Compute the satellite position at a given time using interpolation.

        Args:
            t (pd.Timestamp | datetime): Time of epoch.
            metadata (pd.Series): Metadata of the SP3 orbit file.
            data (pd.DataFrame): DataFrame of the SP3 orbit file. (See SP3 Parser)
            tolerance (int): Tolerance in sp3 extra interpolation.
            kwargs: Additional keyword arguments.

        Returns:
            pd.Series: Series of the satellite position (x, y, z coordinates)
                       interpolated at the given time.

        Raises:
            ValueError: If essential data (x, y, z coordinates or time) is missing
                        or if the given time is outside the time range of the data.
        """
        # Check if x, y, z are available in columns
        # and 'time' is available in index
        if not all([x in data.columns for x in ["x", "y", "z"]]):
            raise ValueError("Missing x, y or z in data")

        if "time" != data.index.name:
            raise ValueError("Missing time in index")

        # Check that the time is between the first and last time
        if not (
            data.index.min() - pd.Timedelta(minutes=tolerance)
            <= t
            <= data.index.max() + pd.Timedelta(minutes=tolerance)
        ):
            raise ValueError(
                f"Time {t} is outside the time range of the data: Time range: {data.index.min()} - {data.index.max()}"
            )

        # Get the nearest subset of the data
        subset = self._nearestSubset(data, t)

        # Initial time of the subset
        t0 = subset.index[0]

        # Time difference between the subset and the given time
        subset["time_elapsed"] = (subset.index.to_series()).apply(
            lambda x: (x - t0).total_seconds()
        )

        # Fit a 11 degree polynomial to the x,y,z data
        x = subset["x"]
        y = subset["y"]
        z = subset["z"]
        t_x = subset["time_elapsed"]

        # Fit a 11 degree polynomial to the x,y,z data
        x_poly = Polynomial.fit(t_x, x, 10)
        y_poly = Polynomial.fit(t_x, y, 10)
        z_poly = Polynomial.fit(t_x, z, 10)
        # Return the interpolated position
        return pd.Series(
            {
                "x": x_poly((t - t0).total_seconds()),
                "y": y_poly((t - t0).total_seconds()),
                "z": z_poly((t - t0).total_seconds()),
            }
        )

    def compute(
        self,
        t: pd.Timestamp | datetime,
        metadata: pd.Series,  # noqa: ARG002
        data: pd.DataFrame,
        tolerance: int = 5,
        **kwargs,
    ) -> pd.Series:
        """Compute the satellite position at a given time using interpolation.

        Args:
            t (pd.Timestamp | datetime): Time of epoch.
            metadata (pd.Series): Metadata of the SP3 orbit file.
            data (pd.DataFrame): DataFrame of the SP3 orbit file. (See SP3 Parser)
            tolerance (int): Tolerance in sp3 extra interpolation.
            kwargs: Additional keyword arguments.

        Returns:
            pd.Series: Series of the satellite position (x, y, z coordinates)
                       interpolated at the given time.

        Raises:
            ValueError: If essential data (x, y, z coordinates or time) is missing
                        or if the given time is outside the time range of the data.
        """
        return self._compute(t, metadata=None, data=data, tolerance=tolerance, **kwargs)
