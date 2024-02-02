"""This is the parse module for Navigator.

This module provides classes for parsing data related to GNSS navigation. The main class is `Parser`, and the interface
classes are in the `iparse` module.

Design Pattern:
    - Builder: The `Parser` class is a builder class for the `IParse` interface.

Interface Available:
    - IParse: An abstract interface for parsing GNSS navigation data.
    - IParseGPSNav: A concrete interface for parsing GPS navigation data.
    - IParseGPSObs: A concrete interface for parsing GPS observation data.
    - IParseSP3GPS: A concrete interface for parsing GPS SP3 data.

Example Usage:
    >>> from navigator.parse import Parser, IParseGPSNav
    >>> parser = Parser(interface=IParseGPSNav())
    >>> parser.parse(filename=navigation_file_path)

Note:
    To parse data, instantiate the `Parser` class and call the `parse` method.

See Also:
    - `navigator.parse.iparse`: The interface module for parsing GNSS navigation data.

References:
    - Georinex: `https://pypi.org/project/georinex/`

Todo:
    - Migrate the backend from `georinex` to `georust` with `pyo3`.
    - Add support for other GNSS systems. (Already done in `georust`)

Examples:
    >>> # Add usage examples here.

"""

from .base_parse import Parser
from .iparse import IParse, IParseGPSNav, IParseGPSObs, IParseSP3GPS
