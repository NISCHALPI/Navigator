"""This module contains the function to convert geocentric coordinates to ellipsoidal coordinates."""

import numpy as np  # type: ignore
import pyproj

__all__ = ["geocentric_to_ellipsoidal", "ellipsoidal_to_geocentric"]


def geocentric_to_ellipsoidal(x: float, y: float, z: float) -> tuple:
    """Convert geocentric coordinates to ellipsoidal coordinates.

    Args:
        x (float): The x-coordinate in geocentric coordinates.
        y (float): The y-coordinate in geocentric coordinates.
        z (float): The z-coordinate in geocentric coordinates.

    Returns:
        tuple: A tuple containing the latitude, longitude, and height in ellipsoidal coordinates.
    """
    # Create a transformer for the conversion
    transformer = pyproj.Transformer.from_crs(
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
    )

    # Perform the transformation
    lon, lat, height = transformer.transform(x, y, z, radians=False)

    # Return the result as a tuple
    return lat, lon, height


def ellipsoidal_to_geocentric(lat: float, lon: float, height: float) -> tuple:
    """Convert ellipsoidal coordinates to geocentric coordinates.

    Args:
        lat (float): The latitude in ellipsoidal coordinates.
        lon (float): The longitude in ellipsoidal coordinates.
        height (float): The height in ellipsoidal coordinates.

    Returns:
        tuple: A tuple containing the x, y, and z coordinates in geocentric coordinates.
    """
    # Create a transformer for the conversion
    transformer = pyproj.Transformer.from_crs(
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
    )

    # Perform the transformation
    x, y, z = transformer.transform(lon, lat, height, radians=False)

    # Return the result as a tuple
    return x, y, z


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
