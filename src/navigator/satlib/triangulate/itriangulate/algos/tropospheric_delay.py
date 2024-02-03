"""This module contains functions to calculate tropospheric delay for the triangulation algorithm.

The tropospheric delay model implemented in this module accounts for signal delay due to the presence of the troposphere. The delay is a function of the elevation angle of the satellite and the distance between the satellite and the receiver. The Saastamoinen model is employed for calculating the delay.

Source:
    - https://gssc.esa.int/navipedia//index.php/Tropospheric_Delay#cite_ref-3

Attributes:
    __all__ (List[str]): List of public symbols to be exported when using "from module import *".

Functions:
    - tropospheric_delay_correction: Calculate the tropospheric delay in the signal.
    - neil_obliquity_factors: Calculate the mapping function for the tropospheric delay using Neil mapping function.
    - common_mapping_function: Calculate the obliquity factor for the tropospheric delay using collins mapping function.


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
    from tropospheric_model import tropospheric_delay_correction

    elevation = 30.0
    height = 100.0
    day_of_year = 180
    delay = tropospheric_delay_correction(elevation, height, day_of_year)
    print(f"Tropospheric Delay: {delay} meters")
    ```
"""

__all__ = ['tropospheric_delay_correction']

from pathlib import Path

import numpy as np
import pandas as pd

# CONSTANTS
k1 = 77.604
k2 = 382000
Rd = 287.054
gm = 9.784
g = 9.80665

# Constants for Neil mapping function
a_ht = 2.53e-5
b_ht = 5.49e-3
c_ht = 1.14e-3


def common_mapping_function(
    elevation: float,
) -> float:
    """Calculate the obliquity factor for the tropospheric delay using collins mapping function.

    Note: This mapping function is valid for elevation angles greater than 5 degrees.

    Args:
        elevation (float): The elevation angle of the satellite in degrees.

    Returns:
        float: The mapping function for the tropospheric delay.
    """
    if elevation < 5:
        elevation = 5
    return 1.001 / (0.002001 + np.sin(np.radians(elevation)) ** 2)


def _neil_mapping_function(
    elevation: float,
    a: float,
    b: float,
    c: float,
) -> float:
    """Calculate the obliquity factor for the tropospheric delay using Neil mapping function.

    Args:
        elevation (float): The elevation angle of the satellite in degrees.
        a (float): The parameter a for the mapping function.
        b (float): The parameter b for the mapping function.
        c (float): The parameter c for the mapping function.

    Returns:
        float: The mapping function for the tropospheric delay.
    """
    E = np.deg2rad(elevation)
    numerator = 1 + (a / (1 + b / (1 + c)))
    denominator = np.sin(E) + (a / (np.sin(E) + b / (np.sin(E) + c)))
    return numerator / denominator


def neil_obliquity_factors(
    elevation: float,
    height: float,
    day_of_year: int,
    hemisphere: bool = True,
) -> tuple[float, float]:
    """This function calculates the mapping function for the tropospheric delay using Neil mapping function.

    Args:
        elevation (float): The elevation angle of the satellite in degrees.
        height (float): The height of the receiver above the sea level in meters.
        day_of_year (int): The day of the year. [1-365]
        hemisphere (bool, optional): The hemisphere. Defaults to True meaning northern hemisphere else southern hemisphere.


    Returns:
        tuple[float, float]: The mapping function for the tropospheric delay. (M_dry, M_wet)

    """
    # Get the parameters for hydrostatic delay
    a_d, b_d, c_d = _get_neil_mapping_parameters_for_hydrostatic_delay(
        elevation, day_of_year, hemisphere
    )
    # Get the parameters for wet delay
    a_w, b_w, c_w = _get_neil_mapping_parameters_for_wet_delay(
        elevation, day_of_year, hemisphere
    )

    # Calculate the mapping function for hydrostatic delay
    E = np.deg2rad(elevation)
    # Calculate delay for dry component
    dm = ((1 / np.sin(E)) - _neil_mapping_function(elevation, a_ht, b_ht, c_ht)) * (
        height / 1000
    )  # Height in km
    M_dry = _neil_mapping_function(elevation, a_d, b_d, c_d) + dm

    # Calculate delay for wet component
    M_wet = _neil_mapping_function(elevation, a_w, b_w, c_w)

    return M_dry, M_wet


def _get_neil_mapping_parameters_for_hydrostatic_delay(
    elevation: float,
    day_of_year: int,
    hemisphere: bool = True,
) -> tuple[float, float, float]:
    """Get the interpolated parameters for the Neil mapping function for hydrostatic delay.

    Args:
        elevation (float): The elevation angle of the satellite in degrees.
        day_of_year (int): The day of the year. [1-365]
        hemisphere (bool, optional): The hemisphere. Defaults to True meaning northern hemisphere.


    Returns:
        tuple[float, float, float: The interpolated parameters for the Neil mapping function for hydrostatic delay.
        (a_d, b_d, c_d)
    """
    # Load the average parameters and variation parameters
    neil_avg_parameters = pd.read_csv(
        Path(__file__).parent
        / "static/neil_mapping/neil_hydrostatic_average_parameters.csv",
        index_col=0,
    )
    neil_var_parameters = pd.read_csv(
        Path(__file__).parent
        / "static/neil_mapping/neil_hydrostatic_variation_parameters.csv",
        index_col=0,
    )

    # Interpolate the parameters for the given elevation
    interpolated_avg = _interpolate_closest_two_elevation_index(
        elevation, neil_avg_parameters
    )
    interpolated_var = _interpolate_closest_two_elevation_index(
        elevation, neil_var_parameters
    )

    return _neil_dispatch_values(
        avg_series=interpolated_avg,
        var_series=interpolated_var,
        day_of_year=day_of_year,
        hemisphere=hemisphere,
        mode="dry",
    )


def _get_neil_mapping_parameters_for_wet_delay(
    elevation: float,
    day_of_year: int,
    hemisphere: bool = True,
) -> tuple[float, float, float, float, float, float]:
    """Get the interpolated parameters for the Neil mapping function for wet delay.

    Args:
        elevation (float): The elevation angle of the satellite in degrees.
        day_of_year (int): The day of the year. [1-365]
        hemisphere (bool, optional): The hemisphere. Defaults to True meaning northern hemisphere.

    Returns:
        tuple[float, float, float,]: The interpolated parameters for the Neil mapping function for wet delay.
        (a_w, b_w, c_w)

    """
    # Load just the average parameters for wet delay since it is not time dependent
    neil_avg_parameters = pd.read_csv(
        Path(__file__).parent / "static/neil_mapping/neil_wet_average_parameters.csv",
        index_col=0,
    )

    # Interpolate the parameters for the given elevation
    interpolated_avg = _interpolate_closest_two_elevation_index(
        elevation, neil_avg_parameters
    )

    return _neil_dispatch_values(
        avg_series=interpolated_avg,
        var_series=pd.Series(),
        day_of_year=day_of_year,
        hemisphere=hemisphere,
        mode="wet",
    )


def _neil_dispatch_values(
    avg_series: pd.Series,
    var_series: pd.Series,
    day_of_year: int,
    hemisphere: bool = True,
    mode: str = "dry",
) -> tuple:
    """Dispatch the values for the given day of the year.

    Args:
        avg_series (pd.Series): The average series.
        var_series (pd.Series): The variation series.
        day_of_year (int): The day of the year. [1-365]
        hemisphere (bool, optional): The hemisphere. Defaults to True meaning northern hemisphere.
        mode (str, optional): The mode of the function. Defaults to "dry".

    Returns:
        tuple: The interpolated a, b, and c values.
    """
    a, b, c = 0, 0, 0
    if mode == "dry":
        # Dry parameters need to be interpolated since they are time dependent
        a = _day_interpolator(
            day_of_year, avg_series["a"], var_series["da"], hemisphere
        )
        b = _day_interpolator(
            day_of_year, avg_series["b"], var_series["db"], hemisphere
        )
        c = _day_interpolator(
            day_of_year, avg_series["c"], var_series["dc"], hemisphere
        )
    else:
        # Wet parameters need not be interpolated
        a = avg_series["a"]
        b = avg_series["b"]
        c = avg_series["c"]

    return a, b, c


def tropospheric_delay_correction(
    elevation: float,
    height: float,
    day_of_year: int,
    hemisphere: bool = True,
    mapping_function: str = "neil",
) -> float:
    """Calculate the tropospheric delay in the signal.

    Args:
        elevation (float): The elevation angle of the satellite in degrees.
        height (float): The height of the receiver above the sea level in meters.
        day_of_year (int): The day of the year. [1-365]
        hemisphere (bool, optional): The hemisphere. Defaults to True meaning northern hemisphere else southern hemisphere.
        mapping_function (str, optional): The mapping function to be used. [collins, neil] Defaults to collins.

    Returns:
        float: The tropospheric delay in the signal in meters.
    """
    # Get the average parameters
    P, T, e, beta, lmda = _get_interpolated_parameters_saastamoinen(
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

    # Calculate the mapping function for the tropospheric delay
    if mapping_function == "collins":
        return common_mapping_function(elevation) * (T_z_dry + T_z_wet)
    #
    if mapping_function == "neil":
        M_dry, M_wet = neil_obliquity_factors(
            elevation, height, day_of_year, hemisphere
        )
        return M_dry * T_z_dry + M_wet * T_z_wet

    raise ValueError("Invalid mapping function. Choose from [collins, neil]")


def _get_interpolated_parameters_saastamoinen(
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
    average_parameters = pd.read_csv(
        fp / "static/tropospheric_parameters/average_parameters.csv", index_col=0
    )
    variation_parameters = pd.read_csv(
        fp / "static/tropospheric_parameters/varaition_parameters.csv", index_col=0
    )

    # Interpolate the parameters for the given elevation
    interpolated_avg = _interpolate_closest_two_elevation_index(
        elevation, average_parameters
    )
    # Interpolate the variation parameters for the given elevation
    interpolated_var = _interpolate_closest_two_elevation_index(
        elevation, variation_parameters
    )

    return _dispatch_values(
        avg_series=interpolated_avg,
        var_series=interpolated_var,
        day_of_year=day_of_year,
        hemisphere=hemisphere,
    )


def _interpolate_closest_two_elevation_index(
    elevation: float,
    frame: pd.DataFrame,
    threash_low: float = 15,
    threash_high: float = 75,
) -> pd.Series:
    """Interpolate the closest two index for given elevation.

    Args:
        elevation (float): The elevation angle of the satellite in degrees.
        frame (pd.DataFrame): The dataframe to interpolate that must have elevation as index.
        threash_low (float, optional): Lower threashold for elevation. Defaults to 15.
        threash_high (float, optional): Higher threashold for elevation. Defaults to 75.

    Returns:
        pd.Series: The average parameters for the given elevation.
    """
    # Get the closest two index for given elevation
    closest = ((frame.index.to_series() - elevation) ** 2).nsmallest(2).index

    # Check if the elevation is out of range [15, 75]
    if elevation < threash_low:
        return frame.loc[closest[0]]
    if elevation > threash_high:
        return frame.loc[closest[1]]

    # Interpolate the values for the given elevation
    start_elv = closest[0]
    end_elv = closest[1]

    # Interpolated average values
    return frame.loc[closest[0]] + (
        (frame.loc[closest[1]] - frame.loc[closest[0]])
        * (elevation - start_elv)
        / (end_elv - start_elv)
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
    """Interpolate the value for the given day of the year using cosine interpolation.

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
