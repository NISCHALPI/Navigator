"""This module contains functions to calculate tropospheric delay for the triangulation algorithm.

The tropospheric delay model implemented in this module accounts for signal delay due to the presence of the troposphere. The delay is a function of the elevation angle of the satellite and the distance between the satellite and the receiver. The Saastamoinen model is employed for calculating the delay.

Source:
    - https://gssc.esa.int/navipedia//index.php/Tropospheric_Delay#cite_ref-3

Attributes:
    __all__ (List[str]): List of public symbols to be exported when using "from module import *".

Functions:
    - tropospheric_delay: Calculate the tropospheric delay in the signal.
    - mapping_function: Calculate the obliquity factor for the tropospheric delay.
    - _get_interpolated_parameters: Get the interpolated parameters for the Saastamoinen model.
    - _dispatch_values: Dispatch the values for the given day of the year.
    - _day_interpolator: Interpolate the value for the given day of the year.

Constants:
    - k1 (float): Constant used in the calculation.
    - k2 (float): Constant used in the calculation.
    - Rd (float): Constant used in the calculation.
    - gm (float): Constant used in the calculation.
    - g (float): Constant used in the calculation.

Note:
    This module provides tools for tropospheric delay calculations, including functions for calculating delay, mapping functions, and interpolation of Saastamoinen model parameters. The constants used in the calculations are also defined at the module level.

Example:
    Usage of the tropospheric_delay function:

    ```python
    from tropospheric_model import tropospheric_delay

    elevation = 30.0
    height = 100.0
    day_of_year = 180
    delay = tropospheric_delay(elevation, height, day_of_year)
    print(f"Tropospheric Delay: {delay} meters")
    ```
"""

__all__ = ['tropospheric_delay']

from pathlib import Path

import numpy as np
import pandas as pd

# CONSTANTS
k1 = 77.604
k2 = 382000
Rd = 287.054
gm = 9.784
g = 9.80665


def mapping_function(
    elevation: float,
) -> float:
    """Calculate the obliquity factor for the tropospheric delay.

    Note: This mapping function is valid for elevation angles greater than 5 degrees.

    Args:
        elevation (float): The elevation angle of the satellite in degrees.

    Returns:
        float: The mapping function for the tropospheric delay.
    """
    if elevation < 5:
        elevation = 5
    return 1.001 / (0.002001 + np.sin(np.radians(elevation)) ** 2)


def tropospheric_delay(
    elevation: float,
    height: float,
    day_of_year: int,
    hemisphere: bool = True,
) -> float:
    """Calculate the tropospheric delay in the signal.

    Args:
        elevation (float): The elevation angle of the satellite in degrees.
        height (float): The height of the receiver above the sea level in meters.
        day_of_year (int): The day of the year. [1-365]
        hemisphere (bool, optional): The hemisphere. Defaults to True meaning northern hemisphere else southern hemisphere.

    Returns:
        float: The tropospheric delay in the signal in meters.
    """
    # Get the average parameters
    P, T, e, beta, lmda = _get_interpolated_parameters(
        elevation, day_of_year, hemisphere
    )

    # Calculate the Zero-altitude vertical delay term
    T_z0_dry = 1e-6 * k1 * Rd * P / gm
    T_z0_wet = (1e-6 * k2 * Rd * e / ((lmda + 1) * gm - beta * Rd)) * (e / T)

    # Calculate the delay at the height of the receiver
    T_z_dry = T_z0_dry * (1 - ((beta * height) / T)) ** (gm / (Rd * beta))
    T_z_wet = T_z0_wet * (1 - ((beta * height) / T)) ** (
        ((lmda + 1) * g / (Rd * beta)) - 1
    )
    # Return the tropospheric delay
    return mapping_function(elevation) * (T_z_dry + T_z_wet)


def _get_interpolated_parameters(
    elevation: float, day_of_year: int, hemisphere: bool = True
) -> tuple:
    """Get the interpolated parameters for the Saastamoinen model.

    Args:
        elevation (float): The elevation angle of the satellite in degrees.
        day_of_year (int): The day of the year. [1-365]
        hemisphere (bool, optional): The hemisphere. Defaults to True meaning northern hemisphere.

    Returns:
        tuple: The interpolated P, T, e, beta, and lmda values.
    """
    fp = Path(__file__).parent
    # Read the average parameters and variation parameters from dataframes
    average_parameters = pd.read_csv(fp / "static/average_parameters.csv", index_col=0)
    variation_parameters = pd.read_csv(
        fp / "static/varaition_parameters.csv", index_col=0
    )

    # Get the closest two index for given elevation
    closest = (
        ((average_parameters.index.to_series() - elevation) ** 2).nsmallest(2).index
    )

    # Get the closest two index for given day of year
    closest_avg = average_parameters.loc[closest]
    closest_variations = variation_parameters.loc[closest]

    # Check if the elevation is out of range [15, 75]
    if elevation < 15:
        return _dispatch_values(
            avg_series=closest_avg.iloc[0],
            var_series=closest_variations.iloc[0],
            day_of_year=day_of_year,
            hemisphere=hemisphere,
        )
    elif elevation > 75:
        return _dispatch_values(
            avg_series=closest_avg.iloc[1],
            var_series=closest_variations.iloc[1],
            day_of_year=day_of_year,
            hemisphere=hemisphere,
        )
    else:
        # Interpolate the values for the given elevation
        start_elv = closest_avg.index[0]
        end_elv = closest_avg.index[1]

        # Interpolated average values
        interpolated_avg = closest_avg.iloc[0] + (
            (closest_avg.iloc[1] - closest_avg.iloc[0])
            * (elevation - start_elv)
            / (end_elv - start_elv)
        )
        # Interpolated variation values
        interpolated_var = closest_variations.iloc[0] + (
            (closest_variations.iloc[1] - closest_variations.iloc[0])
            * (elevation - start_elv)
            / (end_elv - start_elv)
        )
        return _dispatch_values(
            avg_series=interpolated_avg,
            var_series=interpolated_var,
            day_of_year=day_of_year,
            hemisphere=hemisphere,
        )


def _dispatch_values(
    avg_series: pd.Series,
    var_series: pd.Series,
    day_of_year: int,
    hemisphere: bool = True,
) -> tuple:
    """Dispatch the values for the given day of the year.

    Args:
        avg_series (pd.Series): The average series.
        var_series (pd.Series): The variation series.
        day_of_year (int): The day of the year. [1-365]
        hemisphere (bool, optional): The hemisphere. Defaults to True meaning northern hemisphere.

    Returns:
        tuple: The interpolated P, T, e, beta, and lmda values.
    """
    # Get the interpolated values
    P = _day_interpolator(day_of_year, avg_series["P"], var_series["dP"], hemisphere)
    T = _day_interpolator(day_of_year, avg_series["T"], var_series["dT"], hemisphere)
    e = _day_interpolator(day_of_year, avg_series["e"], var_series["de"], hemisphere)
    beta = _day_interpolator(
        day_of_year, avg_series["beta"], var_series["dbeta"], hemisphere
    )
    lmda = _day_interpolator(
        day_of_year, avg_series["lmda"], var_series["dlmda"], hemisphere
    )

    return P, T, e, beta, lmda


def _day_interpolator(
    day_of_year: int, val: float, variation: float, hemisphere: bool = True
) -> float:
    """Interpolate the value for the given day of the year.

    Args:
        day_of_year (int): The day of the year. [1-365]
        val (float): The average value.
        variation (float): The variation value.
        hemisphere (bool, optional): The hemisphere. Defaults to True meaning northern hemisphere.

    Returns:
        float: The interpolated value for the given day of the year.
    """
    # Check if in northern hemisphere or southern hemisphere
    D_min = 28 if hemisphere else 211

    # Interpolate the value
    return val + variation * np.cos(2 * np.pi * (day_of_year - D_min) / 365.25)
