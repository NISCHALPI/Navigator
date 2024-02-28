"""This module contains the function to convert geocentric coordinates to ellipsoidal coordinates."""

import numba as nb
import numpy as np  # type: ignore

__all__ = ["geocentric_to_ellipsoidal", "ellipsoidal_to_geocentric"]


@nb.njit(nb.float64[:](nb.float64, nb.float64, nb.float64), fastmath=True, cache=True)
def geocentric_to_ellipsoidal(
    x: np.float64, y: np.float64, z: np.float64
) -> np.ndarray:
    """Convert geocentric coordinates to ellipsoidal coordinates.

    Args:
        x (np.float64): x-coordinate in meters
        y (np.float64): y-coordinate in meters
        z (np.float64): z-coordinate in meters

    Returns:
        np.ndarray: Ellipsoidal coordinates
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563  # flattening

    # Calculate longitude and latitude
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - f))

    # Iteratively refine latitude using the iterative method
    tolerance = 1e-9
    while True:
        previous_lat = lat
        N = a / np.sqrt(1 - f * (2 - f) * np.sin(lat) ** 2)
        lat = np.arctan2(z + f * N * np.sin(lat), p)

        # Break the loop if the change in latitude is small enough
        if np.abs(lat - previous_lat) < tolerance:
            break

    # Calculate height
    height = p / np.cos(lat) - N

    # Convert latitude and longitude from radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    # Normalize longitude to be between -180 and 180
    lon = (lon + 180) % 360 - 180

    # Normalize latitude to be between -90 and 90
    lat = (lat + 90) % 180 - 90

    return np.array([lat, lon, height])


@nb.njit(
    nb.float64[:](nb.float64, nb.float64, nb.float64),
    fastmath=True,
    error_model="numpy",
    cache=True,
)
def ellipsoidal_to_geocentric(
    lat: np.float64, lon: np.float64, alt: np.float64
) -> np.ndarray:
    """Convert geocentric coordinates to ellipsoidal coordinates.

    Args:
        lat (np.float64): Latitude in degrees
        lon (np.float64): Longitude in degrees
        alt (np.float64): Altitude in meters

    Returns:
        np.ndarray: Ellipsoidal coordinates
    """
    # WG84 ellipsoid parameters
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = 1 - (b**2 / a**2)
    # Convert degrees to radians
    lat = np.radians(lat)
    lon = np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)
    return np.array([x, y, z])


def ellipsoidal_to_enu(
    lat: float, long: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ellipsoidal coordinates to ENU coordinates.

    Args:
        lat (float): The latitude in ellipsoidal coordinates. Units: degrees
        long (float): The longitude in ellipsoidal coordinates. Units: degrees

    Returns:
        tuple: Unit vectors for the east, north, and up directions.
    """
    # Convert the latitude and longitude to radians
    lat_rad = np.deg2rad(lat)
    long_rad = np.deg2rad(long)

    e_hat = np.array([-np.sin(long_rad), np.cos(long_rad), 0])
    n_hat = np.array(
        [
            -np.sin(lat_rad) * np.cos(long_rad),
            -np.sin(lat_rad) * np.sin(long_rad),
            np.cos(lat_rad),
        ]
    )
    u_hat = np.array(
        [
            np.cos(lat_rad) * np.cos(long_rad),
            np.cos(lat_rad) * np.sin(long_rad),
            np.sin(lat_rad),
        ]
    )

    return e_hat, n_hat, u_hat


def geocentric_to_enu(
    x: float, y: float, z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geocentric coordinates to ENU coordinates.

    Args:
        x (float): The x-coordinate in geocentric coordinates.
        y (float): The y-coordinate in geocentric coordinates.
        z (float): The z-coordinate in geocentric coordinates.

    Returns:
        tuple: Unit vectors for the east, north, and up directions.
    """
    lat, long, _ = geocentric_to_ellipsoidal(x, y, z)
    return ellipsoidal_to_enu(lat, long)
