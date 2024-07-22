"""UNB3 Tropospheric Model.

This module provides the implementation of the UNB3m tropospheric model
used to calculate the zenith hydrostatic and wet delays based on the
latitude, day of the year, and height of the receiver above sea level.

References:
    - https://www.ion.org/publications/abstract.cfm?articleID=6562
    - https://www.sciencedirect.com/science/article/pii/S0263224112003879#b0080
"""

import numpy as np

from .seasonal_atmospheric_model import get_seasonal_atmospheric_values

# Define Constants
k1 = 77.604
k2 = 16.6
k3 = 377600
R = 287.054
g = 9.80665

__all__ = ["UNB3m"]


class UNB3m:
    """UNB3m tropospheric model to calculate the zenith delays.

    This class implements the UNB3m tropospheric model which calculates
    the zenith hydrostatic and wet delays using the provided latitude,
    day of the year, and height of the receiver.

    Attributes:
        None

    Methods:
        get_zenith_delays(latitude, day_of_year, height):
            Calculates the zenith hydrostatic and wet delays.
    """

    def get_zenith_delays(
        self,
        latitude: float,
        day_of_year: int,
        height: float,
    ) -> tuple[float, float]:
        """Calculate the zenith delays.

        This method calculates the zenith hydrostatic and wet delays based
        on the latitude, day of the year, and height of the receiver above
        sea level using the UNB3m tropospheric model.

        Args:
            latitude (float): The latitude of the receiver.
            day_of_year (int): The day of the year.
            height (float): The height of the receiver above sea level.

        Returns:
            tuple[float, float]: The zenith hydrostatic and wet delays.
        """
        # Get the seasonal atmospheric values
        P = get_seasonal_atmospheric_values(
            day_of_year=day_of_year, latitude=latitude, key="P"
        )
        T = get_seasonal_atmospheric_values(
            day_of_year=day_of_year, latitude=latitude, key="T"
        )
        beta = get_seasonal_atmospheric_values(
            day_of_year=day_of_year, latitude=latitude, key="beta"
        )
        lmda = get_seasonal_atmospheric_values(
            day_of_year=day_of_year, latitude=latitude, key="lmda"
        )
        RH = get_seasonal_atmospheric_values(
            day_of_year=day_of_year, latitude=latitude, key="RH"
        )

        # Convert the relative humidity to a water vapor pressure
        # Calculate the saturation vapor pressure
        e_s = 0.01 * np.exp(
            1.2378847e-5 * T**2 - 1.9121316e-2 * T + 33.93711047 - 6.3431645e3 / T
        )
        # Calculate the enhancement factor
        f_w = 1.00062 + 3.14e-6 * P - 5.6e-7 * (T - 273.15) ** 2
        # Calculate the water vapor pressure
        e_0 = RH * f_w * e_s / 100

        # Calculate the g_m factor
        g_m = 9.784 * (1 - 2.66e-3 * np.cos(2 * latitude) - 2.8e-7 * height)

        # Calculate the zenith hydrostatic delay
        factor = 1 - beta * height / T
        factor = np.clip(factor, 0, None)
        d_hz = (1e-6 * k1 * R / g_m) * P * factor ** (g / (R * beta))

        # Calculate the zenith wet delay
        lmda_prime = lmda + 1
        T_m = T * (1 - (beta * R / (g_m * lmda_prime)))

        d_nh = (
            1e-6
            * (T_m * k2 + k3)
            * R
            / (g_m * lmda_prime - beta * R)
            * (e_0 / T)
            * factor ** ((lmda_prime * g / (R * beta)) - 1)
        )

        return d_hz, d_nh
