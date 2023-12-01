"""These are rotation matrix corssponding to the transformation from the Celestial Reference System (CRS) to the Terrestrial Reference System (TRS).

References:
Chapter 3.1.2.2 of the following book by ESA:
https://server.gage.upc.edu/TEACHING_MATERIAL/GNSS_Book/ESA_GNSS-Book_TM-23_Vol_I.pdf#page=58&zoom=100,262,174

"""


import numpy as np


def R1(theta: float) -> np.ndarray:
    """Rotation matrix around the x-axis.

    Args:
        theta (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotation matrix.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)],
        ]
    )


def R2(theta: float) -> np.ndarray:
    """Rotation matrix around the y-axis.

    Args:
        theta (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ]
    )


def R3(theta: float) -> np.ndarray:
    """Rotation matrix around the z-axis.

    Args:
        theta (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
