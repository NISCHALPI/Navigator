"""This module provides an implementation of the Klobuchar ionospheric model, offering a function to compute the ionospheric delay for single-frequency GNSS receivers. The Klobuchar model is a mathematical representation of the Earth's ionosphere's impact on signals from GNSS satellites.

Functions:
    - klobuchar_ionospheric_model(latitude, longitude, E, A, t, ionospheric_parameters, frequency=1.57542e9)
        Computes the ionospheric delay for a given set of input parameters.

Args:
            latitude (float): The latitude of the receiver in degrees.
            longitude (float): The longitude of the receiver in degrees.
            E (float): The elevation angle of the satellite in degrees.
            A (float): The azimuth angle of the satellite in degrees.
            t (float): The current time in seconds.
            ionospheric_parameters (dict): The ionospheric parameters.
            frequency (float, optional): The frequency of the signal in Hz for which the ionospheric delay is to be computed. Default is for L1 frequency (1.57542e9 Hz).

Returns:
            float: The ionospheric delay in meters.

Source:
    The Klobuchar ionospheric model is based on information available at:
    - [Klobuchar Ionospheric Model](https://gssc.esa.int/navipedia/index.php?title=Klobuchar_Ionospheric_Model)

Note:
    The ionospheric delay is an important correction factor for accurate positioning in GNSS receivers, compensating for the delay introduced by the Earth's ionosphere. This module facilitates the application of the Klobuchar model for precise GNSS signal processing.
"""

import numpy as np

__all__ = ['klobuchar_ionospheric_model']

L1_FREQUENCY = 1.57542e9  # Hz


def klobuchar_ionospheric_model(
    latitude: float,
    longitude: float,
    E: float,
    A: float,
    t: float,
    ionospheric_parameters: dict,
    frequency: float = 1.57542e9,
) -> float:
    """Klobuchar ionospheric model for single-frequency GNSS receivers.

    Args:
        latitude (float): The latitude of the receiver in degrees.
        longitude (float): The longitude of the receiver in degrees.
        E (float): The elevation angle of the satellite in degrees.
        A (float): The azimuth angle of the satellite in degrees.
        t (float): The current time in seconds.
        ionospheric_parameters (dict): The ionospheric parameters.
        frequency (float): The frequency of the signal in Hz for which the ionospheric delay is to be computed. Default is for L1 frequency (1.57542e9 Hz).

    Returns:
        The ionospheric delay in meters.
    """
    # Convert angles to radians
    phi_u = np.deg2rad(latitude)
    lamda_u = np.deg2rad(longitude)
    E = np.deg2rad(E)
    A = np.deg2rad(A)

    # Compute the Earth-centered angle
    psi = 0.0137 / (E + 0.11) - 0.022

    # Compute the latitude of the Ionospheric Pierce Point (IPP)
    phi_i = phi_u + psi * np.cos(A)
    # Ensure that phi_i is within the range [-0.416, 0.416]
    if phi_i > 0.416:
        phi_i = 0.416
    elif phi_i < -0.416:
        phi_i = -0.416

    # Compute the longitude of the IPP.
    lamda_i = lamda_u + psi * np.sin(A) / np.cos(phi_i)
    #  Find the geomagnetic latitude of the IPP.
    phi_m = phi_i + 0.064 * np.cos(lamda_i - 1.617)

    # Compute the local time at the IPP.
    t = 43200 * lamda_i + t
    # Ensure that t is within the range [0, 86400]
    t = t % 86400
    if t < 0:
        t += 86400

    # Compute the amplitude of the ionospheric delay
    alpha0 = ionospheric_parameters['alpha0']
    alpha1 = ionospheric_parameters['alpha1']
    alpha2 = ionospheric_parameters['alpha2']
    alpha3 = ionospheric_parameters['alpha3']
    A = alpha0 + alpha1 * phi_m + alpha2 * phi_m**2 + alpha3 * phi_m**3
    # Ensure that A is greater than 0.0
    if A < 0.0:
        A = 0.0

    # Compute the period of the ionospheric delay
    beta0 = ionospheric_parameters['beta0']
    beta1 = ionospheric_parameters['beta1']
    beta2 = ionospheric_parameters['beta2']
    beta3 = ionospheric_parameters['beta3']
    P = beta0 + beta1 * phi_m + beta2 * phi_m**2 + beta3 * phi_m**3
    # Ensure that P is greater than 72000
    if P < 72000:
        P = 72000

    # Compute the phase of the ionospheric delay
    X_I = 2 * np.pi * (t - 50400) / P

    #  Compute the slant factor
    F = 1.0 + 16.0 * (0.53 - E) ** 3

    iono_delay = 0.0
    # Compute the ionospheric delay
    if np.abs(X_I) >= np.pi / 2:
        iono_delay = 5.0e-9 * F
    else:
        iono_delay = (5.0e-9 + A * (1 - (X_I**2) / 2 + (X_I**4) / 24)) * F

    # Return the ionospheric delay in meters for the given frequency
    return (L1_FREQUENCY / frequency) ** 2 * iono_delay
