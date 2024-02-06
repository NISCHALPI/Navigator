"""This module contains functions to calculate tropospheric delay for the triangulation algorithm.

The tropospheric delay is the delay in the signal due to the presence of the troposphere. The troposphere is the lowest layer of the Earth's atmosphere. It is the layer in which we live and breathe. It is the densest layer of the atmosphere and contains approximately 80% of the mass of the atmosphere. The troposphere is the layer where most of the weather occurs. The troposphere extends from the Earth's surface to an average height of about 12 km. The troposphere is characterized by a decrease in temperature with height, and the presence of water vapor.

Source:
    - https://gssc.esa.int/navipedia//index.php/Tropospheric_Delay#cite_ref-3

Attributes:
    __all__ (List[str]): List of public symbols to be exported when using "from module import *".

Functions:
    tropospheric_delay_correction: Calculate the tropospheric delay in the signal using neil mapping function.
"""

import numpy as np

from .egnos_tropospheric_correction_model import EgnosTroposphericModel
from .neil_mapping import NeilMapping

__all__ = ['tropospheric_delay_correction']


def tropospheric_delay_correction(
    latitude: float,
    elevation: float,
    height: float,
    day_of_year: int,
) -> float:
    """Calculate the tropospheric delay in the signal.

    Args:
        latitude (float): The latitude of the receiver in degrees. [-90, 90]
        elevation (float): The elevation angle of the satellite in degrees.
        height (float): The height of the receiver above the sea level in meters.
        day_of_year (int): The day of the year. [1-365]

    Returns:
        float: The tropospheric delay in the signal in meters.
    """
    # Get the Zenith Delays using the EGNOS tropospheric model
    Z_dry, Z_wet = EgnosTroposphericModel().get_tropospheric_correction(
        latitude=latitude,
        height=height,
        day_of_year=day_of_year,
        hemisphere=True if latitude >= 0 else False,
    )

    # Get the mapping parameters
    M_dry, M_wet = NeilMapping().get_neil_mapping_parameters(
        latitude=latitude,
        elevation=elevation,
        height=height,
        day_of_year=day_of_year,
        hemisphere=True if latitude >= 0 else False,
    )

    return Z_dry * M_dry + Z_wet * M_wet


def filtered_troposphere_correction(
    latitude: float,
    elevation: float,
    height: float,
    estimated_wet_delay: float,
    day_of_year: int,
) -> float:
    """Calculate the tropospheric delay in the signal if the wet delay is estimated.

    Args:
        latitude (float): The latitude of the receiver in degrees. [-90, 90]
        elevation (float): The elevation angle of the satellite in degrees.
        height (float): The height of the receiver above the sea level in meters.
        estimated_wet_delay (float): The estimated wet delay in the signal in meters from the Kalman filter.
        day_of_year (int): The day of the year. [1-365]

    Returns:
        float: The tropospheric delay in the signal in meters.

    Source:
        - https://gssc.esa.int/navipedia//index.php/Tropospheric_Delay#cite_ref-3
    """
    # Define the constants
    alpha = 2.3
    beta = 0.116e-3

    # Clip Height for initial satbility
    height = np.clip(height, -6000, 6000)

    # Get the dry delay
    Tr_dry = alpha * np.exp(-beta * height)
    Tr_wet = 0.1 + estimated_wet_delay

    # Calculate the neil mapping function
    M_dry, M_wet = NeilMapping().get_neil_mapping_parameters(
        latitude=latitude,
        elevation=elevation,
        height=height,
        day_of_year=day_of_year,
        hemisphere=True if latitude >= 0 else False,
    )

    return Tr_dry * M_dry + Tr_wet * M_wet
