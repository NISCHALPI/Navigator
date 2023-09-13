"""This module implements the IGPSEphemeris class.

The `IGPSEphemeris` class is designed to calculate the position of a satellite by utilizing
ephemeris data from RINEX (Receiver Independent Exchange Format) navigation files. This class
serves as an interface for computing satellite positions and is intended to be used as an
attachment to the Satellite class.

Ephemeris data is essential for determining the precise orbital information of a satellite,
including its position, velocity, and clock corrections. This class provides a foundation for
implementing the necessary calculations and data processing to derive satellite positions.

Note that the '_compute' method within this class is a placeholder for your specific
implementation, which should follow the guidelines and specifications outlined in the ICD
(Interface Control Document) or other relevant documentation.

Attributes:
    None

Methods:
    - _compute(metadata: pd.Series, data: pd.Series) -> pd.Series:
      This method should be implemented to calculate the satellite's position based on
      ephemeris data and return a Series containing the calculated position information.

Usage:
    To use the `IGPSEphemeris` class, inherit from it and provide a concrete implementation
    of the '_compute' method. This implementation should adhere to the requirements specified
    in the documentation for the satellite navigation system you are working with.

Example:
    # Create a concrete implementation of IGPSEphemeris
    class MyEphemeris(IGPSEphemeris):
        def _compute(self, metadata: pd.Series, data: pd.Series) -> pd.Series:
            # Implement the position calculation based on ephemeris data
            # Return a Series containing the calculated position information
            pass

    # Instantiate the MyEphemeris class and use it as an attachment to the Satellite class.
    ephemeris = MyEphemeris()
    satellite = Satellite(iephemeris=ephemeris)
    positions = satellite(metadata=metadata, data=data)

Note:
    The actual implementation of the '_compute' method should follow the specifications
    provided by the satellite navigation system's documentation.

"""

import numpy as np
import pandas as pd

from ..iephm import AbstractIephemeris

__all__ = ["IGPSEphemeris"]


# Sources
# https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf
# https://ascelibrary.org/doi/pdf/10.1061/9780784411506.ap03



# Constants
F = -4.442807633e-10  # Constant for computing relativistic clock correction
gps_start_time = pd.Timestamp('1980-01-06 00:00:00')  # GPS start time
gps_week_seconds = 604800  # Number of seconds in a GPS week
GM = 3.986005e14  # Earth's universal gravitational parameter
omega_e = 7.2921151467e-5  # Earth's rotation rate


class IGPSEphemeris(AbstractIephemeris):
    """Interface for calculating satellite positions using RINEX ephemeris data.

    Subclass this interface and implement the '_compute' method for specific satellite systems.
    """

    def __init__(self) -> None:
        """Initialize an IGPSEphemeris instance."""
        super().__init__(feature="GPS")

    def _relativistic_clock_correction(self, sqrt_A: float, Ek: float) -> float:
        """Calculate the relativistic clock correction.

        Args:
            sqrt_A (pd.Series): Square root of the semi-major axis of the orbit.
            Ek (pd.Series): Keplers eccentric anomaly.
        """
        # See : https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf page 98
        return F * np.exp(sqrt_A) * np.sin(Ek)

    def _week_anamoly(self, time_diff: float) -> float:
        """Accounts for week crossovers.

        Args:
            time_diff (float): time difference in seconds.

        Returns:
            float: time difference in seconds accounting for week crossovers.
        """
        if time_diff > 302400:
            time_diff -= 604800
        elif time_diff < -302400:
            time_diff += 604800

        return time_diff

    def _eccentric_anomaly(
        self, t_k: float, sqrt_A: float, deltaN: float, M_o: float, e: float
    ) -> float:
        """Calculate the eccentric anomaly.

        Args:
            t_k (float): Time from ephemeris reference epoch.
            sqrt_A (float): Square root of the semi-major axis of the orbit.
            deltaN (float): Mean motion difference from computed value.
            M_o (float): Mean anomaly at reference time.
            e (float): Eccentricity of the orbit.


        Returns:
            float: Eccentric anomaly.
        """
        # See : https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf page 106

        # Semi-major axis of the orbit
        A = sqrt_A**2

        # Computed mean motion
        n_o = np.sqrt(GM / A**3)

        # Corrected mean motion
        n = n_o + deltaN

        # Mean anomaly
        M_k = M_o + n * t_k

        # Iteratively solve for eccentric anomaly
        E_k1 = M_k
        E_k2 = 0

        # Iteratively solve for eccentric anomaly
        while True:
            E_k2 = M_k + e * np.sin(E_k1)
            if abs(E_k2 - E_k1) < 1e-12:
                break
            E_k1 = E_k2

        return E_k2

    def _gps_week_to_datetime(self, week: int, seconds: float) -> pd.Timestamp:
        """Convert GPS week and seconds to datetime."""
        return gps_start_time + pd.Timedelta(weeks=week, seconds=seconds)

    def _clock_correction(self, metadata: pd.Series, data: pd.Series) -> pd.Timedelta:
        """Calculate the clock correction. See GPS ICD 200 Page 98: https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf X.

        Args:
            metadata (pd.Series): Metadata of the RINEX file.
            data (pd.Series): Data for specific satellite.

        Returns:
            pd.Series: _description_
        """
        # Sv clock correction rates
        a_f0 = data['SVclockBias']
        a_f1 = data['SVclockDrift']
        a_f2 = data['SVclockDriftRate']

        # Time from ephemeris reference epoch
        t_oc: pd.Timestamp = data['Toc']

        # SV time
        t_sv: pd.Timestamp = metadata['Tsv']

        # Toe
        t_oe = self._gps_week_to_datetime(data['GPSWeek'], data['Toe'])

        # Compute Ek usinge pre-corrected time
        Ek = self._eccentric_anomaly(
            t_k=self._week_anamoly((t_sv - t_oe).seconds),
            sqrt_A=data['sqrtA'],
            deltaN=data['deltaN'],
            M_o=data['M0'],
            e=data['e'],
        )

        # Get Relativitic clock correction
        t_r = self._relativistic_clock_correction(
            sqrt_A=data['sqrtA'],
            Ek=Ek,
        )

        # Group delay differential
        t_gd = data['TGD']

        # Compute clock correction
        delta_t = self._week_anamoly(
            t=(t_sv - t_oc).seconds,
        )
        # Compute clock correction
        return pd.Timedelta(
            seconds=a_f0 + a_f1 * delta_t + a_f2 * delta_t**2 - t_gd + t_r
        )

    def _compute(self, metadata: pd.Series, data: pd.Series) -> pd.Series:
        """Calculate the satellite's position based on ephemeris data.

        Args:
            metadata (pd.Series): Metadata of the RINEX file.
            data (pd.Series): Data for specific satellite.

        Returns:
            pd.Series: Return a Series containing the calculated position information [x, y , z] in WGS84-ECFC coordinates.
        """
        # Get clock correction for the satellite time i.e. Tsv
        t : pd.Timestamp = data['Tsv'] - self._clock_correction(metadata=metadata, data=data)

        # Get the time difference from the ephemeris reference epoch
        t_k = self._week_anamoly((t - self._gps_week_to_datetime(data['GPSWeek'], data['Toe'])).seconds)

        # Get the eccentric anomaly
        Ek = self._eccentric_anomaly( # noqa
            t_k=t_k,
            sqrt_A=data['sqrtA'],
            deltaN=data['deltaN'],
            M_o=data['M0'],
            e=data['e'],
        )

        # Get the true anomaly
        vk = np.arctan2(
            y=np.sqrt(1 - data['e']**2) * np.sin(Ek),
            x=np.cos(Ek) - data['e'],
        )
        
        # Get the argument of latitude
        phik = vk + data['omega']
        # TO DO : Complete the implementation of the _compute method
        pass
