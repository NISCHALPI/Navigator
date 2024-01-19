"""This is the satlib module for Navigator, serving as the primary module for GNSS data processing.

Submodules:
    - `satellite`: Handles satellite data processing.
    - `triangulate`: Manages user-end data processing for triangulation.

Example Usage:
    >>> from navigator.satlib import Triangulate, IterativeTriangulationInterface
    >>> triangulator = Triangulate(interface=IterativeTriangulationInterface())
    >>> triangulator.process(obs=obs_epoch, nav_metadata=nav_metadata, obs_metadata=obs_metadata)

Note:
    This module acts as a super module for the `satellite` and `triangulate` modules. Refer to the respective modules for detailed information.

See Also:
    - `navigator.satlib.satellite`: Handles satellite data processing.
    - `navigator.satlib.triangulate`: Manages user-end data processing for triangulation.
"""
from .satellite import IGPSEphemeris, IGPSSp3, Satellite
from .triangulate import (
    IterativeTriangulationInterface,
    Triangulate,
    UnscentedKalmanTriangulationInterface,
)
