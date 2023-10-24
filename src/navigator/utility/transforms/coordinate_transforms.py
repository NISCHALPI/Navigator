"""This module contains the function to convert geocentric coordinates to ellipsoidal coordinates."""
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
