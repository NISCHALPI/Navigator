"""This module contains functions to calculate tropospheric delay for the triangulation algorithm.

The tropospheric delay is the delay in the signal due to the presence of the troposphere. The troposphere is the lowest layer of the Earth's atmosphere. It is the layer in which we live and breathe. It is the densest layer of the atmosphere and contains approximately 80% of the mass of the atmosphere. The troposphere is the layer where most of the weather occurs. The troposphere extends from the Earth's surface to an average height of about 12 km. The troposphere is characterized by a decrease in temperature with height, and the presence of water vapor.

Source:
    - https://gssc.esa.int/navipedia//index.php/Tropospheric_Delay#cite_ref-3

Attributes:
    __all__ (List[str]): List of public symbols to be exported when using "from module import *".

Functions:
    tropospheric_delay_correction: Calculate the tropospheric delay in the signal using neil mapping function.
"""

from .neil_mapping import NeilMapping
from .saastamoinen_model import SaastamoinenTroposphericModel

__all__ = [
    "tropospheic_delay_with_neil_map",
    "saastamoinen_tropospheric_correction_with_neil_mapping",
]


def tropospheic_delay_with_neil_map(
    ZHD: float,
    ZWD: float,
    latitude: float,
    elevation: float,
    height: float,
    day_of_year: int,
) -> float:
    """Calculate the tropospheric delay in the signal.

    Args:
        ZHD (float): The zenith hydrostatic delay in meters.
        ZWD (float): The zenith wet delay in meters.
        latitude (float): The latitude of the receiver in degrees. [-90, 90]
        elevation (float): The elevation angle of the satellite in degrees.
        height (float): The height of the receiver above the sea level in meters.
        day_of_year (int): The day of the year. [1-365]

    Returns:
        float: The tropospheric delay in the signal in meters.
    """
    # Get the mapping parameters
    M_dry, M_wet = NeilMapping().get_neil_mapping_parameters(
        latitude=latitude,
        elevation=elevation,
        height=height,
        day_of_year=day_of_year,
        hemisphere=True if latitude >= 0 else False,
    )
    return ZHD * M_dry + ZWD * M_wet


def saastamoinen_tropospheric_correction_with_neil_mapping(
    latitude_of_receiver: float,
    elevation_angle_of_satellite: float,
    height_of_receiver: float,
    day_of_year: int,
) -> float:
    """Calculate the slant tropospheric delay using the Saastamoinen tropospheric model with Neil mapping function.

    Args:
        latitude_of_receiver (float): The latitude of the receiver in degrees. [-90, 90]
        elevation_angle_of_satellite (float): The elevation angle of the satellite in degrees.
        height_of_receiver (float): The height of the receiver above the sea level in meters.
        day_of_year (int): The day of the year. [1-365]

    Returns:
        float: The slant tropospheric delay in meters.
    """
    # Get the zenith hydrostatic delay  and zenith wet delay
    ZHD, ZWD = SaastamoinenTroposphericModel().get_zenith_delays(
        height=height_of_receiver,
        latitude=latitude_of_receiver,
    )

    return tropospheic_delay_with_neil_map(
        ZHD=ZHD,
        ZWD=ZWD,
        latitude=latitude_of_receiver,
        elevation=elevation_angle_of_satellite,
        height=height_of_receiver,
        day_of_year=day_of_year,
    )
