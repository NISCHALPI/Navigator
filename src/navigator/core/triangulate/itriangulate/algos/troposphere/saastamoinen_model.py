"""Implements the Saastamoinen tropospheric model for GPS/GNSS signals.

References:
    - Atmospheric Correction for the Troposphere and Stratosphere in Radio Ranging Satellites
        by Saastamoinen, J. 1972
"""

import numpy as np

__all__ = ["SaastamoinenTroposphericModel"]


class SaastamoinenTroposphericModel:
    """Initializes the Saastamoinen tropospheric correction model."""

    STD_TEMP = 291.15
    STD_PRESS = 1013.25
    STD_HUMIDITY = 50

    def atmospheric_condition_at(
        self,
        height: float,
    ) -> tuple[float, float, float]:
        """Returns the atmospheric conditions at a given height using standard atmosphere model.

        Args:
            height (float): The height in meters above sea level.

        Returns:
            tuple(float, float, float): The temperature, pressure, and humidity at the given height.
        """
        p = self.STD_PRESS * (1 - 0.0000226 * height) ** 5.225
        temp = self.STD_TEMP - 0.0065 * height
        humidity = self.STD_HUMIDITY * np.exp(-0.0006396 * height)

        return temp, p, humidity

    def ZHD(
        self,
        pressure: float,
        height: float,
        latitude: float,
    ) -> float:
        """Calculates the zenith hydrostatic delay (ZHD) using the Saastamoinen tropospheric model.

        Args:
            pressure (float): The pressure in hPa at the receiver location.
            height (float): The height in meters above sea level of the receiver.
            latitude (float): The latitude in degrees of the receiver in degrees.

        Returns:
            float: The zenith hydrostatic delay in meters.
        """
        return (
            0.0022767
            * pressure
            * (1 + 0.00266 * np.cos(2 * np.radians(latitude)) - 0.00000028 * height)
        )

    def ZWD(
        self,
        humidity: float,
        temperature: float,
    ) -> float:
        """Calculates the zenith wet delay (ZWD) using the Saastamoinen tropospheric model.

        Args:
            humidity (float): The relative humidity in percentage at the receiver location.
            temperature (float): The temperature in Kelvin at the receiver location.
            height (float): The height in meters above sea level of the receiver.

        Returns:
            float: The zenith wet delay in meters.
        """
        # Convert humidity to fraction
        humidity /= 100

        c = -37.2465 + 0.213166 * temperature - 2.56908 * (10e-4) * (temperature**2)
        e = humidity * np.exp(c)  # Water vapor pressure in hPa

        return 0.0022768 * ((1255 / temperature) + 0.05) * e

    def get_zenith_delays(
        self,
        height: float,
        latitude: float,
    ) -> tuple[float, float]:
        """Returns the zenith hydrostatic and wet delays at a given height.

        Args:
            height (float): The height in meters above sea level.
            latitude (float): The latitude in degrees of the receiver in degrees.

        Returns:
            tuple(float, float): The zenith hydrostatic and wet delays in meters.
        """
        # Height must be in [0, 5000] meters
        height = np.clip(height, 0, 5000)
        temp, pressure, humidity = self.atmospheric_condition_at(height)
        zhd = self.ZHD(
            pressure=pressure,
            height=height,
            latitude=latitude,
        )
        zwd = self.ZWD(
            humidity=humidity,
            temperature=temp,
        )

        return zhd, zwd
