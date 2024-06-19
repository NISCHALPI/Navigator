"""Factory module to create constellations based on the name.

This module provides functions to create constellations of satellites based on different criteria.

Attributes:
    unit_cube (numpy.ndarray): Coordinates of the vertices of a unit cube used for creating a constellation.

Functions:
    get_unit_cube_constellation: Create a constellation with 8 stationary trajectories at the vertices of a unit cube.
    get_gps_constellation: Create a GPS constellation with the given number of satellites.

Constants:
    unit_cube (numpy.ndarray): Coordinates of the vertices of a unit cube used for creating a constellation.
"""

import typing as tp
from pathlib import Path

import numpy as np

from ..trajectory import StationaryTrajectory
from ..trajectory.gps import from_almanac
from .constellation import Constellation

__all__ = ["get_unit_cube_constellation", "get_gps_constellation"]
unit_cube = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
) - np.array([0.5, 0.5, 0.5])


def get_unit_cube_constellation() -> Constellation:
    """Create a constellation with 8 stationery trajectories at the vertices of a unit cube.

    Returns:
        Constellation: A constellation object with 8 stationery trajectories at the vertices of a unit cube.
    """
    trajectories = {
        f"T{i}": StationaryTrajectory(
            x=unit_cube[i, 0], y=unit_cube[i, 1], z=unit_cube[i, 2]
        )
        for i in range(unit_cube.shape[0])
    }

    return Constellation(trajectories=trajectories)


def get_gps_constellation(
    num_gps: int = 31, almanac: tp.Optional[Path] = None
) -> Constellation:
    """Create a GPS constellation with the given number of satellites.

    Args:
        num_gps (int, optional): The number of GPS satellites to create. Defaults to 31.
        almanac (tp.Optional[Path], optional): The path to the almanac file. Defaults to None.

    Returns:
        Constellation: A GPS constellation object with the given number of satellites.
    """
    # Get Kepelerian satellites from the almanac
    sv = from_almanac(almanac=almanac)

    # Filter SVs based on the number of GPS satellites
    sv = {k: sv[k] for k in list(sv.keys())[:num_gps]}

    return Constellation(trajectories=sv)
