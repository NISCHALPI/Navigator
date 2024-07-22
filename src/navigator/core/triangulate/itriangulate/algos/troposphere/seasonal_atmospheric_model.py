"""The seasonal atmospheric model from U.S Standard Atmosphere Supplements."""

import numpy as np
import pandas as pd

__all__ = ["get_seasonal_atmospheric_values", "get_seasonal_value_at", "get_average_at"]

AVERAGE = pd.DataFrame(
    {
        "P": {15: 1013.25, 30: 1017.25, 45: 1015.75, 60: 1011.75, 75: 1013.0},
        "T": {15: 299.65, 30: 294.15, 45: 283.15, 60: 272.15, 75: 263.65},
        "WVP": {15: 26.31, 30: 21.79, 45: 11.66, 60: 6.78, 75: 4.11},
        "beta": {15: 0.0063, 30: 0.00605, 45: 0.00558, 60: 0.00539, 75: 0.00453},
        "lmda": {15: 2.77, 30: 3.15, 45: 2.57, 60: 1.81, 75: 1.55},
        "RH": {15: 75.0, 30: 80.0, 45: 76.0, 60: 77.5, 75: 82.5},
    }
)


AMPLITUDE = pd.DataFrame(
    {
        "P": {15: 0.0, 30: -3.75, 45: -2.25, 60: -1.75, 75: -0.5},
        "T": {15: 0.0, 30: 7.0, 45: 11.0, 60: 15.0, 75: 14.5},
        "WVP": {15: 0.0, 30: 8.85, 45: 7.24, 60: 5.36, 75: 3.39},
        "beta": {15: 0.0, 30: 0.00025, 45: 0.00032, 60: 0.00081, 75: 0.00062},
        "lmda": {15: 0.0, 30: 0.33, 45: 0.46, 60: 0.74, 75: 0.3},
        "RH": {15: 0.0, 30: 0.0, 45: -1.0, 60: -2.5, 75: 2.5},
    }
)


def get_average_at(data: pd.DataFrame, key: str, latitude: float) -> float:
    """Perform the linear interpolation to get the average value at the given latitude.

    Args:
        data (pd.DataFrame): The data to interpolate.
        key (str): The key to interpolate.
        latitude (float): The latitude to interpolate.

    Returns:
        float: The interpolated value.
    """
    if latitude < 15:
        return data[key][15]
    if latitude > 75:
        return data[key][75]

    # Calculate where the data falls between the two values
    i, j = 15, 30
    while j <= 75:
        if latitude >= i and latitude <= j:
            break
        i, j = j, j + 15

    # Perform the linear interpolation
    return data[key][i] + (data[key][j] - data[key][i]) * (latitude - i) / 15


def get_seasonal_value_at(
    average: pd.DataFrame,
    amplitude: pd.DataFrame,
    key: str,
    day_of_year: int,
    latitude: float,
) -> float:
    """Get the seasonal value at the given latitude and day of year.

    Args:
        average (pd.DataFrame): The average values.
        amplitude (pd.DataFrame): The amplitude values.
        key (str): The key to get the value.
        day_of_year (int): The day of year.
        latitude (float): The latitude.

    Returns:
        float: The seasonal value.
    """
    # Calculate the average value
    avg_phi = get_average_at(average, key, abs(latitude))
    amp_phi = get_average_at(amplitude, key, abs(latitude))

    # Adjust for southern hemisphere
    if latitude < 0:
        day_of_year = day_of_year + 182.625

    return avg_phi - (amp_phi * np.cos(2 * np.pi * (day_of_year - 28) / 365.25))


def get_seasonal_atmospheric_values(
    day_of_year: int, latitude: float, key: str
) -> float:
    """Get the seasonal atmospheric value at the given latitude and day of year.

    Args:
        day_of_year (int): The day of year.
        latitude (float): The latitude.
        key (str): The key to get the value.

    Returns:
        float: The seasonal atmospheric value.
    """
    return get_seasonal_value_at(AVERAGE, AMPLITUDE, key, day_of_year, latitude)
