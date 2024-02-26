"""This module contains the EGNOS tropospheric correction model.

The EGNOS tropospheric correction model is a tropospheric correction model that estimates zenith tropospheric delay (ZTD) for a given location and time.

Paper:
    Penna N, Dodson A, Chen W. Assessment of EGNOS Tropospheric Correction Model. Journal of Navigation. 2001;54(1):37-55. doi:10.1017/S0373463300001107

Attributes:
    average_parameters (pd.DataFrame): Average atmospheric parameters for different elevations.
    seasonal_variation (pd.DataFrame): Seasonal variation of atmospheric parameters for different elevations.
    k_1 (float): Constant for dry tropospheric correction calculation.
    k_2 (float): Constant for wet tropospheric correction calculation.
    R_d (float): Specific gas constant for dry air.
    g (float): Standard gravity constant.
    g_m (float): Gravity at mean sea level.
    Dmin_southern (float): Minimum day of the year for Southern Hemisphere variation.
    Dmin_northern (float): Minimum day of the year for Northern Hemisphere variation.

Classes:
    EgnosTroposphericModel: Class implementing the EGNOS tropospheric correction model.

Methods:
    __init__(self): Initializes the EGNOS tropospheric correction model.
    get_tropospheric_correction(latitude, height, day_of_year, hemisphere=True): Gets tropospheric correction for a location and time.
    zenith_dry_delay_at_sea_level(pressure): Computes zenith dry delay at sea level.
    zenith_wet_delay_at_sea_level(temperature, humidity, lmda, beta): Computes zenith wet delay at sea level.
    _get_parameters(latitude, day_of_year, hemisphere=True): Gets average parameters for a given elevation.
    _interpolate_closest_two_elevation_index(elevation, frame, threash_low=15, threash_high=75): Interpolates values for the given elevation.

"""

import numpy as np
import pandas as pd

__all__ = ["EgnosTroposphericModel"]


class EgnosTroposphericModel:
    """The EGNOS tropospheric correction model is a tropospheric correction model that is used to estimate zenith tropospheric delay (ZTD) for a given location and time.

    The EGNOS tropospheric correction model is a tropospheric correction model that is used to estimate zenith tropospheric delay (ZTD) for a given location and time.

    Paper:
        Penna N, Dodson A, Chen W. Assessment of EGNOS Tropospheric Correction Model. Journal of Navigation. 2001;54(1):37-55. doi:10.1017/S0373463300001107

    """

    average_parameters = pd.DataFrame(
        {
            "P": {15: 1013.25, 30: 1017.25, 45: 1015.75, 60: 1011.75, 75: 1013.0},
            "T": {15: 299.65, 30: 294.15, 45: 283.15, 60: 272.15, 75: 263.65},
            "e": {15: 26.31, 30: 21.79, 45: 11.66, 60: 6.78, 75: 4.11},
            "beta": {15: 0.0063, 30: 0.00605, 45: 0.00558, 60: 0.00539, 75: 0.00453},
            "lmda": {15: 2.77, 30: 3.15, 45: 2.57, 60: 1.81, 75: 1.55},
        }
    )

    seasonal_variation = pd.DataFrame(
        {
            "dP": {15: 0.0, 30: -3.75, 45: -2.25, 60: -1.75, 75: -0.5},
            "dT": {15: 0.0, 30: 7.0, 45: 11.0, 60: 15.0, 75: 14.5},
            "de": {15: 0.0, 30: 8.85, 45: 7.24, 60: 5.36, 75: 3.39},
            "dbeta": {15: 0.0, 30: 0.00025, 45: 0.00032, 60: 0.00081, 75: 0.00062},
            "dlmda": {15: 0.0, 30: 0.33, 45: 0.46, 60: 0.74, 75: 0.3},
        }
    )
    # Constants
    k_1 = 77.604
    k_2 = 382000
    R_d = 287.054
    g = 9.80665
    g_m = 9.784
    Dmin_southern = 211.0
    Dmin_northern = 28.0

    def __init__(self) -> None:
        """Initializes the EGNOS tropospheric correction model."""
        return

    def get_tropospheric_correction(
        self,
        latitude: float,
        height: float,
        day_of_year: int,
        hemisphere: bool = True,
    ) -> tuple[float, float]:
        """Get the tropospheric correction for given location and time.

        Args:
            latitude (float): The latitude of the location in degrees.
            height (float): The height of the location in meters.
            day_of_year (int): The day of the year.
            hemisphere (bool, optional): Hemisphere of the location. Defaults to True (Northern Hemisphere).

        Returns:
            tuple[float, float]: The dry and wet tropospheric correction in meters.[Tdry, Twet]
        """
        # Get the average parameters for the given elevation
        avg_parameter = self._get_parameters(latitude, day_of_year, hemisphere)

        # Compute the zenith dry and wet delay at sea level
        Z_dry = self.zenith_dry_delay_at_sea_level(avg_parameter["P"])
        Z_wet = self.zenith_wet_delay_at_sea_level(
            temperature=avg_parameter["T"],
            humidity=avg_parameter["e"],
            lmda=avg_parameter["lmda"],
            beta=avg_parameter["beta"],
        )
        # Clip the height to be greater that -200m
        height = np.clip(height, -200, None)

        # Compute the height correction
        height_factor = 1 - ((avg_parameter["beta"] * height) / avg_parameter["T"])

        # Compute the dry and wet delay at the given height
        dry_pow = self.g / (self.R_d * avg_parameter["beta"])
        wet_pow = ((avg_parameter["lmda"] + 1) * dry_pow) - 1

        # Compute the dry and wet delay at the given height
        d_dry = Z_dry * np.power(height_factor, dry_pow)
        d_wet = Z_wet * np.power(height_factor, wet_pow)
        return d_dry, d_wet

    def zenith_dry_delay_at_sea_level(self, pressure: float) -> float:
        """Compute the zenith dry delay at sea level.

        Args:
            pressure (float): The pressure at sea level in hPa.

        Returns:
            float: The zenith dry delay at sea level in meters.
        """
        numerator = 1e-6 * self.k_1 * self.R_d * pressure
        return numerator / self.g_m

    def zenith_wet_delay_at_sea_level(
        self,
        temperature: float,
        humidity: float,
        lmda: float,
        beta: float,
    ) -> float:
        """Compute the zenith wet delay at sea level.

        Args:
            temperature (float): The temperature at sea level in K.
            humidity (float): The humidity at sea level in %.
            lmda (float): The water vapor lapse rate in K/m.
            beta (float): The temperature lapse rate in K/m.

        Returns:
            float: The zenith wet delay at sea level in meters.
        """
        numerator = 1e-6 * self.k_2 * self.R_d
        denominator = (self.g_m * (lmda + 1)) - (beta * self.R_d)
        factor = humidity / temperature
        return (numerator * factor) / denominator

    def _get_parameters(
        self, latitude: float, day_of_year: int, hemisphere: bool = True
    ) -> pd.Series:
        """Get the average parameters for the given elevation.

        Args:
            latitude (float): The latitude of the location in degrees.
            day_of_year (int): The day of the year.
            hemisphere (bool, optional): Hemisphere of the location. Defaults to True (Northern Hemisphere).

        Returns:
            pd.Series: The average parameters for the given elevation.
        """
        # Compute the average parameters for the given elevation
        avg_parameter = self._interpolate_closest_two_latitude_index(
            latitude, self.average_parameters
        )

        # Compute the seasonal variation for the given elevation
        seasonal_variation = self._interpolate_closest_two_latitude_index(
            latitude, self.seasonal_variation
        )

        # Propagate throught the seasonal variation
        avg_parameter -= seasonal_variation.values * np.cos(
            2
            * np.pi
            * (
                (day_of_year - self.Dmin_northern if hemisphere else self.Dmin_southern)
                / 365.25
            )
        )
        return avg_parameter

    @staticmethod
    def _interpolate_closest_two_latitude_index(
        latitude: float,
        frame: pd.DataFrame,
        threash_low: float = 15,
        threash_high: float = 75,
    ) -> pd.Series:
        """Interpolate the closest two index for given elevation.

        Args:
            latitude (float): The elevation angle of the satellite in degrees.
            frame (pd.DataFrame): The dataframe to interpolate that must have elevation as index.
            threash_low (float, optional): Lower threashold for elevation. Defaults to 15.
            threash_high (float, optional): Higher threashold for elevation. Defaults to 75.

        Returns:
            pd.Series: The average parameters for the given elevation.
        """
        # Get the closest two index for given elevation
        closest = ((frame.index.to_series() - latitude) ** 2).nsmallest(2).index

        # Check if the elevation is out of range [15, 75]
        if latitude < threash_low:
            return frame.loc[closest[0]]
        if latitude > threash_high:
            return frame.loc[closest[1]]

        # Interpolate the values for the given elevation
        start_elv = closest[0]
        end_elv = closest[1]

        t = (latitude - start_elv) / (end_elv - start_elv)
        # Interpolated average values
        return frame.loc[start_elv] * (1 - t) + frame.loc[end_elv] * t
