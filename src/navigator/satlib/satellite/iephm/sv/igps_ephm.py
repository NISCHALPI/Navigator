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


import numpy as np
import pandas as pd

from ..iephm import AbstractIephemeris
from .tools.coord import _eccentric_anomaly, ephm_to_coord_gps, week_anamonaly

__all__ = ["IGPSEphemeris"]


# Constants
gps_start_time = pd.Timestamp(
    year=1980, day=6, month=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0
)
F = -4.442807633e-10  # Constant used in relativistic clock correction


class IGPSEphemeris(AbstractIephemeris):
    """Interface for calculating satellite positions using RINEX ephemeris data.

    Subclass this interface and implement the '_compute' method for specific satellite systems.
    """

    def __init__(self) -> None:
        """Initialize an IGPSEphemeris instance."""
        super().__init__(feature="GPS")

    def _relativistic_clock_correction(
        self, sqrt_A: float, Ek: float, e: float
    ) -> float:
        """Calculate the relativistic clock correction.

        Args:
            sqrt_A (pd.Series): Square root of the semi-major axis of the orbit.
            Ek (pd.Series): Keplers eccentric anomaly.
            e (pd.Series): Eccentricity of the orbit.
        """
        # See : https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf page 98
        return F * e * sqrt_A * np.sin(Ek)

    def _to_seconds_gps_week(self, t: pd.Timestamp, week: int) -> float:
        """Convert a timestamp to seconds of GPS week from current week.

        Args:
            t (pd.Timestamp): Timestamp to convert.
            week (int): GPS week number.


        Returns:
            float: Seconds of GPS week.
        """
        return (t - gps_start_time).total_seconds() - (week * 604800)

    def clock_correction(self, data: pd.Series) -> float:
        """Calculate the clock correction. See GPS ICD 200 Page 98: https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf X.

        Args:
            data (pd.Series): Data for specific satellite.

        Returns:
            float: Clock correction.
        """
        # Sv clock correction rates
        a_f0 = data["SVclockBias"]
        a_f1 = data["SVclockDrift"]
        a_f2 = data["SVclockDriftRate"]

        # Week number
        week = data["GPSWeek"]

        # Time from ephemeris reference epoch
        t_oc: pd.Timestamp = self._to_seconds_gps_week(data["Toc"], week=week)

        # SV time
        t_sv: pd.Timestamp = self._to_seconds_gps_week(data["Tsv"], week=week)

        # Toe
        t_oe = data["Toe"]  # Given in seconds of GPS week

        # Compute Ek usinge pre-corrected time
        Ek = _eccentric_anomaly(
            t_k=week_anamonaly(t=t_sv, t_oe=t_oe),
            sqrt_a=data["sqrtA"],
            delta_n=data["DeltaN"],
            M_0=data["M0"],
            e=data["Eccentricity"],
        )

        # Get Relativitic clock correction
        t_r = self._relativistic_clock_correction(
            sqrt_A=data["sqrtA"],
            Ek=Ek,
            e=data["Eccentricity"],
        )

        # Group delay differential
        t_gd = data["TGD"]

        # Compute delta_t with week anomally
        delta_t = week_anamonaly(t=t_sv, t_oe=t_oc)

        # Compute clock correction
        return a_f0 + a_f1 * delta_t + a_f2 * delta_t**2 + t_r - t_gd

    def _compute(
        self, metadata: pd.Series, data: pd.Series, **kwargs  # noqa: ARG002
    ) -> pd.Series:
        """Calculate the satellite's position based on ephemeris data.

        Args:
            metadata (pd.Series): Metadata of the RINEX file.
            data (pd.Series): Data for specific satellite.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: Return a Series containing the calculated position information [x, y , z] in WGS84-ECFC coordinates.
        """
        # Get clock correction for the satellite time i.e. Tsv
        dt = self.clock_correction(data=data)

        # Correct the satellite time i.e. Tsv
        t_sv: pd.Timestamp = data["Tsv"] - pd.Timedelta(seconds=dt)
        t = self._to_seconds_gps_week(t=t_sv, week=data["GPSWeek"])

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
                "dt": dt,
            }
        )
