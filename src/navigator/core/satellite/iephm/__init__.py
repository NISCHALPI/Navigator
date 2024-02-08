"""This package contains the abstract class for the satellite ephemeris interface and its concrete implementations.

Abstract Class:
    - AbstractIEphemeris (abc.ABC): An abstract class for the satellite ephemeris interface.

Interfaces Available:
    - IGPSEphemeris (AbstractIEphemeris): A concrete class for the GPS ephemeris interface.
    - IGPSSp3 (AbstractIEphemeris): A concrete class for the GPS SP3 interface.

Example Usage:
    >>> from navigator.core import Satellite, IGPSEphemeris
    >>> satellite_processor = Satellite(interface=IGPSEphemeris())
    >>> satellite_processor(filename=nav_dataframe)

Note:
    This interface is not meant to be instantiated directly. Instead, use the `Satellite` class from the `navigator.core` module.

See Also:
    - `navigator.core`: The satlib module for Navigator.
"""

from .iephm import AbstractIephemeris
from .sv.igps_ephm import IGPSEphemeris
from .sv.igps_sp3_orbit import IGPSSp3
