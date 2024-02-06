"""This module contains the Neil Mapping function for tropospheric correction.

This class implements the Neil Mapping function for tropospheric correction.
The Neil Mapping function is used to estimate the tropospheric correction for a given latitude,
longitude, and day of year.

Attributes:
    average_hydrostatic_parameter (pd.DataFrame): DataFrame containing average hydrostatic parameters.
    seasonal_hydrostatic_variation (pd.DataFrame): DataFrame containing seasonal hydrostatic variations.
    average_wet_parameter (pd.DataFrame): DataFrame containing average wet parameters.
    Dmin_southern (float): Minimum day of the year for the southern hemisphere.
    Dmin_northern (float): Minimum day of the year for the northern hemisphere.
    a_ht (float): Height correction parameter a.
    b_ht (float): Height correction parameter b.
    c_ht (float): Height correction parameter c.

Methods:
    __init__(): Constructor for the Neil Mapping function.
    get_neil_mapping_parameters(latitude, elevation, height, day_of_year, hemisphere=True) -> tuple[float, float]:
        Calculates the mapping function for tropospheric delay using the Neil mapping function.
    _mapping_func(elevation, a, b, c) -> float:
        Calculates the mapping function for tropospheric delay.
    _get_dry_parameters(latitude, day_of_year, hemisphere=True) -> pd.Series:
        Calculates the dry parameters for the Neil mapping function.
    _get_wet_parameters(latitude) -> pd.Series:
        Calculates the wet parameters for the Neil mapping function.

"""

import numpy as np
import pandas as pd

from .egnos_tropospheric_correction_model import EgnosTroposphericModel

__all__ = ["NeilMapping"]


class NeilMapping:
    """Neil Mapping function for tropospheric correction.

    This class implements the Neil Mapping function for tropospheric correction.
    The Neil Mapping function is used to estimate the tropospheric correction for a given latitude,
    longitude, and day of year.

    """

    average_hydrostatic_parameter = pd.DataFrame(
        {
            'a': {
                15: 0.0012769934,
                30: 0.001268323,
                45: 0.00124653963,
                60: 0.0012196049,
                75: 0.0012045996,
            },
            'b': {
                15: 0.0029153695,
                30: 0.0029152299,
                45: 0.0029288445,
                60: 0.0029022565,
                75: 0.0029024912,
            },
            'c': {
                15: 0.062610505,
                30: 0.062837393,
                45: 0.063721774,
                60: 0.063824265,
                75: 0.064258455,
            },
        }
    )

    seasonal_hydrostatic_variation = pd.DataFrame(
        {
            'da': {
                15: 0.0,
                30: 1.2709626e-05,
                45: 2.652366e-05,
                60: 3.4000452e-05,
                75: 4.1202191e-05,
            },
            'db': {
                15: 0.0,
                30: 2.1414979e-05,
                45: 3.0160779e-05,
                60: 7.2562722e-05,
                75: 0.00011723375,
            },
            'dc': {
                15: 0.0,
                30: 9.01284e-05,
                45: 4.3497037e-05,
                60: 0.00084795348,
                75: 0.0017037206,
            },
        }
    )

    average_wet_parameter = pd.DataFrame(
        {
            'a': {
                15: 0.00058021897,
                30: 0.00056794847,
                45: 0.00058118019,
                60: 0.00059727542,
                75: 0.00061641693,
            },
            'b': {
                15: 0.0014275268,
                30: 0.0015138625,
                45: 0.001452752,
                60: 0.0015007428,
                75: 0.0017599082,
            },
            'c': {
                15: 0.043472961,
                30: 0.04672951,
                45: 0.043908931,
                60: 0.044626982,
                75: 0.054736038,
            },
        }
    )
    Dmin_southern = 211.0
    Dmin_northern = 28.0

    # Height correction parameters
    a_ht = 2.53e-5
    b_ht = 5.49e-3
    c_ht = 1.14e-3

    def __init__(self) -> None:
        """Constructor for the Neil Mapping function."""
        pass

    def get_neil_mapping_parameters(
        self,
        latitude: float,
        elevation: float,
        height: float,
        day_of_year: int,
        hemisphere: bool = True,
    ) -> tuple[float, float]:
        """This function calculates the mapping function for the tropospheric delay using Neil mapping function.

        Args:
            latitude (float): The latitude of the receiver in degrees.
            elevation (float): The elevation angle of the satellite in degrees.
            height (float): The height of the receiver above the sea level in meters.
            day_of_year (int): The day of the year. [1-365]
            hemisphere (bool, optional): The hemisphere. Defaults to True meaning northern hemisphere else southern hemisphere.


        Returns:
            tuple[float, float]: The mapping function for the tropospheric delay. (M_dry, M_wet)

        """
        # Calculate the dry and wet parameters as per the latitude and day of year
        dry_params = self._get_dry_parameters(latitude, day_of_year)
        wet_params = self._get_wet_parameters(latitude)

        # Calculate the dry mapping function
        dM_dry = (1 / np.sin(np.deg2rad(elevation))) - self._mapping_func(
            elevation, self.a_ht, self.b_ht, self.c_ht
        )
        # Height correction
        dM_dry = dM_dry * (height / 1000)
        M_dry = (
            self._mapping_func(elevation, dry_params.a, dry_params.b, dry_params.c)
            + dM_dry
        )

        # Calculate the wet mapping function
        M_wet = self._mapping_func(elevation, wet_params.a, wet_params.b, wet_params.c)

        return M_dry, M_wet

    def _mapping_func(self, elevation: float, a: float, b: float, c: float) -> float:
        """This function calculates the mapping function for the tropospheric delay.

        Args:
            elevation (float): The elevation angle of the satellite in degrees.
            a (float): The parameter a for the Neil mapping function.
            b (float): The parameter b for the Neil mapping function.
            c (float): The parameter c for the Neil mapping function.

        Returns:
            float: The mapping function for the tropospheric delay.

        """
        numerator = 1 + (a / (1 + b / (1 + c)))
        sin_E = np.sin(np.deg2rad(elevation))
        denominator = sin_E + (a / (sin_E + b / (sin_E + c)))
        return numerator / denominator

    def _get_dry_parameters(
        self, latitude: float, day_of_year: int, hemisphere: bool = True
    ) -> pd.Series:
        """This function calculates the dry parameters for the Neil mapping function.

        Args:
            latitude (float): The latitude of the receiver in degrees.
            day_of_year (int): The day of the year. [1-365]
            hemisphere (bool, optional): The hemisphere. Defaults to True meaning northern hemisphere else southern hemisphere.

        Returns:
            tuple[float, float, float]: The dry parameters for the Neil mapping function. (a, b, c)

        """
        # Get the interpolated values for the latitude
        average_dry = EgnosTroposphericModel._interpolate_closest_two_latitude_index(
            latitude=latitude, frame=self.average_hydrostatic_parameter
        )
        seasonal_dry = EgnosTroposphericModel._interpolate_closest_two_latitude_index(
            latitude=latitude, frame=self.seasonal_hydrostatic_variation
        )

        # Propogate the interpolated values to the day of year
        average_dry -= seasonal_dry.values * np.cos(
            2
            * np.pi
            * (
                (day_of_year - self.Dmin_northern if hemisphere else self.Dmin_southern)
                / 365.25
            )
        )

        return average_dry

    def _get_wet_parameters(self, latitude: float) -> pd.Series:
        """This function calculates the wet parameters for the Neil mapping function.

        Args:
            latitude (float): The latitude of the receiver in degrees.
            day_of_year (int): The day of the year. [1-365]

        Returns:
            pd.Series: The wet parameters for the Neil mapping function. (a, b, c)

        """
        return EgnosTroposphericModel._interpolate_closest_two_latitude_index(
            latitude=latitude, frame=self.average_wet_parameter
        )
