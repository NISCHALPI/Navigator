"""This is the satlib module for Navigator, serving as the primary module for satellite data processing.

Design Pattern:
    - Builder: The `Satellite` class is a builder class for the `AbstractSatellite` interface.

Interface Available:
    - IGPSEphemeris (abc.ABC): An abstract interface for GNSS ephemeris data.
    - IGPSSp3 (abc.ABC): An abstract interface for GNSS SP3 data.

Example Usage:
    >>> from navigator.satlib import Satellite, IGPSEphemeris
    >>> satellite_processor = Satellite(interface=IGPSEphemeris())
    >>> satellite_processor(filename=nav_dataframe)

Note:
    To process satellite data, instantiate the `Satellite` class with an interface and call the `__call__` method.

See Also:
    - `navigator.satlib.satellite.iephm`: The interface module for GNSS ephemeris data.

Todo:
    - Add support for other GNSS systems.
    - Migrate to a Rust backend for performance improvement.
"""
from .iephm import IGPSEphemeris, IGPSSp3
from .satellite import AbstractSatellite, Satellite
