"""Module imports for the constellation package for simulating satellite constellations.

This module provides convenient imports for accessing classes and functions related to satellite constellations.

Modules:
    constellation: Contains the Constellation class for representing a satellite constellation.
    constellation_factory: Contains functions for creating satellite constellations based on various criteria.

Classes:
    Constellation: Represents a satellite constellation.

Functions:
    from_almanac: Extracts satellite trajectory data from an almanac file.
    get_gps_constellation: Creates a GPS satellite constellation.
    get_unit_cube_constellation: Creates a satellite constellation with trajectories at the vertices of a unit cube.
"""

from .constellation import Constellation
from .constellation_factory import (
    from_almanac,
    get_gps_constellation,
    get_unit_cube_constellation,
)
