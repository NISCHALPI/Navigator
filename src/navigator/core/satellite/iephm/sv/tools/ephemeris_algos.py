"""This module contains functions to calculate the coordinate of different satellites.

The procedure is taken from the following GPS ICD 200 document:

Links:
- `GPS ICD 200 Document <https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf>`_
"""

import numpy as np

# Constants
MU = 3.986005e14  # Gravitational constant of the Earth (m^3/s^2)
omega_e = 7.2921151467e-5  # Angular velocity of the Earth (rad/s)
F = -4.442807633e-10  # Constant used in relativistic clock correction

__all__ = [
    "week_anamonaly",
    "eccentric_anomaly",
    "relativistic_clock_correction",
    "clock_correction",
    "ephm_to_coord_gps",
]


def week_anamonaly(t: float, t_oe: float) -> float:
    """Calculate the week anomally of the GPS, Galileo and BeiDou satellites.

    Args:
        t (float): Time of transmission (seconds of GPS week) in satellite time.
        t_oe (float): Time of Ephemeris (seconds of GPS week).


    Returns:
        float: Week anomally of the GPS, Galileo and BeiDou satellites.
    """
    if t - t_oe > 302400:
        return t - t_oe - 604800
    if t - t_oe < -302400:
        return t - t_oe + 604800
    return t - t_oe


# Helper Functions to compute E_k
def eccentric_anomaly(
    t_k: float,
    sqrt_a: float,
    delta_n: float,
    M_0: float,
    e: float,
) -> float:
    """Calculate the eccentric anomaly of the GPS, Galileo and BeiDou satellites.

    Args:
        t_k (float): Time from ephemeris reference epoch (seconds of GPS week).
        sqrt_a (float): Square root of the semi-major axis of the orbit (m^0.5).
        delta_n (float): Mean motion difference from computed value (radians/s).
        M_0 (float): Mean anomaly at reference time (radians).
        e (float): Eccentricity of the orbit.

    Returns:
        float: Eccentric anomaly of the GPS, Galileo and BeiDou satellites.
    """
    # Semi-major axis of the orbit
    a = sqrt_a**2

    # Mk
    M_k = M_0 + (np.sqrt(MU / a**3) + delta_n) * t_k

    # Solve Kepler's equation for E_k
    E_k = M_k

    # Solve Kepler's equation for E_k
    while True:
        E_k_old = E_k
        E_k = E_k_old + (M_k - E_k_old + e * np.sin(E_k_old)) / (
            1 - e * np.cos(E_k_old)
        )
        if np.abs(E_k - E_k_old) < 1e-12:
            break

    return E_k


def relativistic_clock_correction(sqrt_A: float, Ek: float, e: float) -> float:
    """Calculate the relativistic clock correction.

    Args:
        sqrt_A (pd.Series): Square root of the semi-major axis of the orbit.
        Ek (pd.Series): Keplers eccentric anomaly.
        e (pd.Series): Eccentricity of the orbit.
    """
    # See : https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf page 98
    return F * e * sqrt_A * np.sin(Ek)


def clock_correction(
    t: float,
    a_f0: float,
    a_f1: float,
    a_f2: float,
    t_oc: float,
    t_oe: float,
    sqrt_A: float,
    delta_n: float,
    M_0: float,
    e: float,
    t_gd: float,
) -> float:
    """Calculate the clock correction. See GPS ICD 200 Page 98: https://www.gps.gov/technical/icwg/IS-GPS-200N.pdf X.

    All the time must be in seconds of GPS week for the GPS, Galileo and BeiDou satellites.

    Args:
        t (float): Time to compute the clock correction at.
        a_f0 (float): SV clock bias.
        a_f1 (float): SV clock drift.
        a_f2 (float): SV clock drift rate.
        t_oc (float): Time of clock.
        t_oe (float): Time of ephemeris.
        sqrt_A (float): Square root of the semi-major axis of the orbit.
        delta_n (float): Mean motion difference.
        M_0 (float): Mean anomaly at reference time.
        e (float): Eccentricity of the orbit.
        t_gd (float): Group delay differential.

    Returns:
        float: Clock correction.
    """
    # Compute Ek usinge pre-corrected time
    Ek = eccentric_anomaly(
        t_k=week_anamonaly(t=t, t_oe=t_oe),
        sqrt_a=sqrt_A,
        delta_n=delta_n,
        M_0=M_0,
        e=e,
    )

    # Get Relativitic clock correction
    t_r = relativistic_clock_correction(
        sqrt_A=sqrt_A,
        Ek=Ek,
        e=e,
    )

    # Compute delta_t with week anomally
    delta_t = t - t_oc

    # Compute clock correction
    return a_f0 + a_f1 * delta_t + a_f2 * delta_t**2 + t_r - t_gd


# See ICD Page 106
# Works for GPS, Galileo and BeiDou
def ephm_to_coord_gps(
    t: float,
    toe: float,
    sqrt_a: float,
    e: float,
    M_0: float,
    w: float,
    i_0: float,
    omega_0: float,
    delta_n: float,
    i_dot: float,
    omega_dot: float,
    c_uc: float,
    c_us: float,
    c_rc: float,
    c_rs: float,
    c_ic: float,
    c_is: float,
) -> np.ndarray:
    """Calculate the coordinate of the GPS, Galileo and BeiDou satellites given orbital parameters and time and clock corrections.

    Args:
        t (float): Time of transmission (seconds of GPS week) in GPS time. (Corrected for the offset between GPST and GPS time from satellite.)
        toe (float): Time of Ephemeris (seconds of GPS week).
        sqrt_a (float): Square root of the semi-major axis of the orbit (m^0.5).
        e (float): Eccentricity of the orbit.
        M_0 (float): Mean anomaly at reference time (radians).
        w (float): Argument of perigee (radians).
        i_0 (float): Inclination angle at reference time (radians).
        omega_0 (float): Longitude of ascending node at reference time (radians).
        delta_n (float): Mean motion difference from computed value (radians/s).
        i_dot (float): Rate of change of inclination angle (radians/s).
        omega_dot (float): Rate of change of longitude of ascending node (radians/s).
        c_uc (float): Amplitude of the cosine harmonic correction term to the argument of latitude (radians).
        c_us (float): Amplitude of the sine harmonic correction term to the argument of latitude (radians).
        c_rc (float): Amplitude of the cosine harmonic correction term to the orbit radius (m).
        c_rs (float): Amplitude of the sine harmonic correction term to the orbit radius (m).
        c_ic (float): Amplitude of the cosine harmonic correction term to the angle of inclination (radians).
        c_is (float): Amplitude of the sine harmonic correction term to the angle of inclination (radians).

    Returns:
        np.ndarray: Array containing the calculated coordinates of the satellite.

    """
    # Compute t_k from t and t_oe
    t_k = week_anamonaly(t, toe)

    # Compute semi-major axis of the orbit
    a = sqrt_a**2

    # Compute eccentric anomaly
    E_k = eccentric_anomaly(
        t_k=t_k,
        sqrt_a=sqrt_a,
        delta_n=delta_n,
        M_0=M_0,
        e=e,
    )

    # Compute true anomaly
    v_k = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E_k / 2))

    # Compute argument of latitude
    phi_k = v_k + w

    # Second harmonic perturbations
    delta_u_k = c_us * np.sin(2 * phi_k) + c_uc * np.cos(2 * phi_k)
    delta_r_k = c_rs * np.sin(2 * phi_k) + c_rc * np.cos(2 * phi_k)
    delta_i_k = c_is * np.sin(2 * phi_k) + c_ic * np.cos(2 * phi_k)

    # Corrected argument of latitude
    u_k = phi_k + delta_u_k

    # Corrected radius
    r_k = a * (1 - e * np.cos(E_k)) + delta_r_k

    # Corrected inclination
    i_k = i_0 + delta_i_k + i_dot * t_k

    # Positions in orbital plane
    x_k_p = r_k * np.cos(u_k)
    y_k_p = r_k * np.sin(u_k)

    # Corrected longitude of ascending node
    omega_k = omega_0 + (omega_dot - omega_e) * t_k - omega_e * toe

    x_k = x_k_p * np.cos(omega_k) - y_k_p * np.cos(i_k) * np.sin(omega_k)
    y_k = x_k_p * np.sin(omega_k) + y_k_p * np.cos(i_k) * np.cos(omega_k)
    z_k = y_k_p * np.sin(i_k)

    return np.array([x_k, y_k, z_k])
