"""This is the parse interface module for Navigator.

Interface:
    - IParse (abc.ABC): An abstract parse interface.
    - IParseGPSNav (IParse): A concrete GPS navigation parse interface.
    - IParseGPSObs (IParse): A concrete GPS observation parse interface.
    - IParseSP3GPS (IParse): A concrete GPS SP3 parse interface.

Example Usage:
    >>> from navigator.parse import IParseGPSNav, Parser
    >>> parser = Parser(iparser=IParseGPSNav())
    >>> parser.parse(filename=navigation_file_path)

Note:
    These interfaces are not meant to be instantiated directly. Instead, use the `Parser` class from the `navigator.parse` module.

See Also:
    - `navigator.parse.iparse.nav`: The interface module for parsing GNSS navigation data.
    - `navigator.parse.iparse.obs`: The interface module for parsing GNSS observation data.

References:
    - Georinex: [https://pypi.org/project/georinex/](https://pypi.org/project/georinex/)

Todo:
    - Migrate the backend from `georinex` to `georust` with `pyo3`.
    - Add support for other GNSS systems. (Already done in `georust`)
"""

# A note to future developers:
# Design Principle: The python API should be same irrespective of the backend used.
from .base_iparse import IParse
from .nav.iparse_gps_nav import IParseGPSNav
from .obs.iparse_gps_obs import IParseGPSObs
from .sp3.iparse_sp3_gps import IParseSP3GPS
