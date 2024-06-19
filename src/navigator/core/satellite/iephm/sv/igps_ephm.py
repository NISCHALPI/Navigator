"""This module provides an interface, `IGPSEphemeris`, for calculating satellite positions using RINEX ephemeris data for GPS satellites.

The module defines classes and functions:
    - `IGPSEphemeris`: An interface to calculate satellite positions using RINEX ephemeris data. Subclasses can implement the '_compute' method for specific satellite systems.
    - Various helper functions for GPS calculations.

Constants:
    - `gps_start_time`: Timestamp representing the start time of the GPS system.
    - `F`: Constant used in relativistic clock correction.

Classes:
    - `IGPSEphemeris`: Interface for calculating satellite positions using RINEX ephemeris data.

Functions:
    - `_relativistic_clock_correction`: Calculate relativistic clock correction.
    - `_to_seconds_gps_week`: Convert a timestamp to seconds of GPS week from the current week.
    - `clock_correction`: Calculate the clock correction for specific satellite data.
    - `_compute`: Calculate the satellite's position based on ephemeris data.

Usage:
    Instantiate `IGPSEphemeris` and implement the '_compute' method for specific satellite systems.

References:
    - RINEX ephemeris data format.
    - GPS ICD 200 (Interface Control Document) for detailed information about GPS satellite data.

Sources:
   - https://server.gage.upc.edu/TEACHING_MATERIAL/GNSS_Book/ESA_GNSS-Book_TM-23_Vol_I.pdf
   - https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf.
"""

import pandas as pd

from ..iephm import AbstractIephemeris
from .tools.ephemeris_algos import clock_correction, ephm_to_coord_gps

__all__ = ["IGPSEphemeris"]


# Constants
gps_start_time = pd.Timestamp(
    year=1980, day=6, month=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0
)


class IGPSEphemeris(AbstractIephemeris):
    """Interface for calculating satellite positions using RINEX ephemeris data.

    Subclass this interface and implement the '_compute' method for specific satellite systems.

    Algorithms:
        - Calculate the relativistic clock correction.
            Source: `IS-GPS-200N.pdf page 98 <https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf>`_
        - Calculate the clock correction.
            Source: `IS-GPS-200N.pdf page 98 <https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf>`_
        - Calculate the satellite's position based on ephemeris data.
            Source: `IS-GPS-200N.pdf page 104-106 <https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf>`_
    """

    MAX_CLOCK_CORRECTION_ITERATIONS = 20
    SV_CLOCK_BIAS_KEY = "SVclockBias"

    def __init__(self) -> None:
        """Initialize an IGPSEphemeris instance."""
        super().__init__(feature="GPS")

    def _to_seconds_gps_week(self, t: pd.Timestamp, week: int) -> float:
        """Convert a timestamp to seconds of GPS week from current week.

        Args:
            t (pd.Timestamp): Timestamp to convert.
            week (int): GPS week number.


        Returns:
            float: Seconds of GPS week.
        """
        return (t - gps_start_time).total_seconds() - (week * 604800.0)

    def iterative_clock_correction(self, t: float, data: pd.Series) -> float:
        """Calculate the clock correction for specific satellite data.

        Uses iterative method to compute the clock correction to sovle the implicit equation.

        Args:
            t (float): Satellite time in seconds of GPS week.
            data (pd.Series): Data for specific satellite.

        Returns:
            float: Clock correction.
        """
        additional_kwargs = {
            "a_f0": data["SVclockBias"],
            "a_f1": data["SVclockDrift"],
            "a_f2": data["SVclockDriftRate"],
            "t_oc": self._to_seconds_gps_week(t=data["Toc"], week=data["GPSWeek"]),
            "t_oe": data["Toe"],
            "sqrt_A": data["sqrtA"],
            "delta_n": data["DeltaN"],
            "M_0": data["M0"],
            "e": data["Eccentricity"],
            "t_gd": data["TGD"],
        }

        # Initial clock correction
        dt = clock_correction(t=t, **additional_kwargs)

        # Iteratively compute the clock correction
        for _ in range(self.MAX_CLOCK_CORRECTION_ITERATIONS):
            # Compute the satellite time i.e. Tsv
            t -= dt

            # Compute the clock correction
            dt = clock_correction(t=t, **additional_kwargs)

            # If the clock correction is within 1 ns, break
            if abs(dt) < 1e-11:
                break

        return dt

    def _compute(
        self,
        t: pd.Timestamp,
        metadata: pd.Series,  # noqa: ARG002
        data: pd.Series,
        **kwargs,  # noqa: ARG002
    ) -> pd.Series:
        """Calculate the satellite's position based on ephemeris data.

        Args:
            t (pd.Timestamp): The SV time at which to calculate the satellite position.
            metadata (pd.Series): Metadata of the RINEX file.
            data (pd.Series): Data for specific satellite.
            **kwargs: Additional keyword arguments.

        Additional keyword arguments:
            no_clock_correction (bool): If True, do not apply clock correction to the satellite time.

        Returns:
            pd.Series: Return a Series containing the calculated position information [x, y , z] in WGS84-ECFC coordinates.

        """
        t = self._to_seconds_gps_week(
            t=t, week=data["GPSWeek"]
        )  # Convert the system time to seconds of GPS week

        # Get clock correction for the satellite time i.e. Tsv
        dt = self.iterative_clock_correction(t=t, data=data)

        # Apply clock correction to the satellite time
        # Unless the user specifies not to apply clock correction
        # i.e the system time is provided instead of the satellite time
        if not kwargs.get("system_time", False):
            t -= dt

        # Compute the coordinates of the satellite
        coords = ephm_to_coord_gps(
            t=t,
            toe=data["Toe"],
            sqrt_a=data["sqrtA"],
            e=data["Eccentricity"],
            M_0=data["M0"],
            w=data["omega"],
            i_0=data["Io"],
            omega_0=data["Omega0"],
            delta_n=data["DeltaN"],
            i_dot=data["IDOT"],
            omega_dot=data["OmegaDot"],
            c_uc=data["Cuc"],
            c_us=data["Cus"],
            c_ic=data["Cic"],
            c_is=data["Cis"],
            c_rc=data["Crc"],
            c_rs=data["Crs"],
        )
        return pd.Series(
            {
                "x": coords[0],
                "y": coords[1],
                "z": coords[2],
                self.SV_CLOCK_BIAS_KEY: dt,
            }
        )
