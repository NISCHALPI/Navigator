"""This module provides an implementation of the Klobuchar ionospheric model, offering a function to compute the ionospheric delay for single-frequency GNSS receivers. The Klobuchar model is a mathematical representation of the Earth's ionosphere's impact on signals from GNSS satellites.

Functions:
    - klobuchar_ionospheric_model(latitude, longitude, E, A, t, ionospheric_parameters, frequency=1.57542e9)
        Computes the ionospheric delay for a given set of input parameters.

Args:
    latitude (float): The latitude of the receiver in degrees.
    longitude (float): The longitude of the receiver in degrees.
    elev (float): The elevation angle of the satellite in degrees.
    azimuth (float): The azimuth angle of the satellite in degrees.
    tow (float): The current time in seconds.
    alpha0 (float): The alpha0 parameter of the Klobuchar model.
    alpha1 (float): The alpha1 parameter of the Klobuchar model.
    alpha2 (float): The alpha2 parameter of the Klobuchar model.
    alpha3 (float): The alpha3 parameter of the Klobuchar model.
    beta0 (float): The beta0 parameter of the Klobuchar model.
    beta1 (float): The beta1 parameter of the Klobuchar model.
    beta2 (float): The beta2 parameter of the Klobuchar model.
    beta3 (float): The beta3 parameter of the Klobuchar model.
    frequency (float, optional): The frequency of the signal in Hz for which the ionospheric delay is to be computed. Default is for L1 frequency (1.57542e9 Hz).

Returns:
            float: The ionospheric delay in meters.

Source:
    The Klobuchar ionospheric model is based on information available at:
    - [Klobuchar Ionospheric Model](https://gssc.esa.int/navipedia/index.php?title=Klobuchar_Ionospheric_Model)

Note:
    The ionospheric delay is an important correction factor for accurate positioning in GNSS receivers, compensating for the delay introduced by the Earth's ionosphere. This module facilitates the application of the Klobuchar model for precise GNSS signal processing.
"""

import numba as nb
import numpy as np

__all__ = ['klobuchar_ionospheric_correction']

# CONSTANTS
L1_FREQUENCY = 1.57542e9  # Hz
c = 2.99792458e8  # speed of light

# CONVERSION FACTORS
deg2semi = 1.0 / 180.0  # degrees to semisircles
semi2rad = np.pi  # semisircles to radians
deg2rad = np.pi / 180.0  # degrees to radians


@nb.njit(
    fastmath=True,
    error_model="numpy",
    boundscheck=True,
    cache=True,
)
def klobuchar_ionospheric_correction(
    latitude: float,
    longitude: float,
    elev: float,
    azimuth: float,
    tow: float,
    alpha0: float,
    alpha1: float,
    alpha2: float,
    alpha3: float,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    frequency: float = 1.57542e9,
) -> float:
    """Computes the ionospheric delay for a given set of input parameters.

    Args:
        latitude (float): The latitude of the receiver in degrees.
        longitude (float): The longitude of the receiver in degrees.
        elev (float): The elevation angle of the satellite in degrees.
        azimuth (float): The azimuth angle of the satellite in degrees.
        tow (float): The current time in seconds.
        alpha0 (float): The alpha0 parameter of the Klobuchar model.
        alpha1 (float): The alpha1 parameter of the Klobuchar model.
        alpha2 (float): The alpha2 parameter of the Klobuchar model.
        alpha3 (float): The alpha3 parameter of the Klobuchar model.
        beta0 (float): The beta0 parameter of the Klobuchar model.
        beta1 (float): The beta1 parameter of the Klobuchar model.
        beta2 (float): The beta2 parameter of the Klobuchar model.
        beta3 (float): The beta3 parameter of the Klobuchar model.
        frequency (float, optional): The frequency of the signal in Hz for which the ionospheric delay is to be computed. Default is for L1 frequency (1.57542e9 Hz).


    Returns:
        float: The ionospheric delay in meters.
    """
    a = azimuth * deg2rad  # azimuth in radians
    e = elev * deg2semi  # elevation angle in semicircles

    psi = 0.0137 / (e + 0.11) - 0.022  # Earth Centered angle

    lat_i = latitude * deg2semi + psi * np.cos(a)  # Subionospheric lat
    if lat_i > 0.416:
        lat_i = 0.416
    elif lat_i < -0.416:
        lat_i = -0.416

    # Subionospheric long
    long_i = longitude * deg2semi + (psi * np.sin(a) / np.cos(lat_i * semi2rad))

    # Geomagnetic latitude
    lat_m = lat_i + 0.064 * np.cos((long_i - 1.617) * semi2rad)

    t = 4.32e4 * long_i + tow
    t = t % 86400.0  # Seconds of day
    if t > 86400.0:
        t -= 86400.0
    elif t < 0.0:
        t += 86400.0

    sF = 1.0 + 16.0 * (0.53 - e) ** 3  # Slant factor

    # Period of model
    PER = beta0 + beta1 * lat_m + beta2 * lat_m**2 + beta3 * lat_m**3
    if PER < 72000.0:
        PER = 72000.0

    x = 2.0 * np.pi * (t - 50400.0) / PER  # Phase of the model

    # Amplitude of the model
    AMP = alpha0 + alpha1 * lat_m + alpha2 * lat_m**2 + alpha3 * lat_m**3
    if AMP < 0.0:
        AMP = 0.0

    # Ionospheric correction
    if abs(x) > 1.57:
        dIon1 = sF * (5.0e-9)
    else:
        dIon1 = sF * (5.0e-9 + AMP * (1.0 - x**2 / 2.0 + x**4 / 24.0))

    return c * dIon1 * (frequency / L1_FREQUENCY) ** 2  # Ionospheric delay (meters)
