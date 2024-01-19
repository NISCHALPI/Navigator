"""Download module for Navigator.

This module provides classes for downloading data related to navigation.

Users are recommended to directly use the download interface module `idownload` since the main download class is not implemented.

Design Pattern:
    - Inheritance: The download interface inherits from the abstract base class `IDownload`.

Interface Available:
    - NasaCddisV3: A class for downloading RINEX files from the NASA CDDIS.
    - NasaCddisIgsSp3: A class for downloading SP3 files from the NASA CDDIS.

State:
    Main class not implemented since the download API differs for different data sources.

Example Usage:
    >>> from navigator.download import NasaCddisV3
    >>> downloader = NasaCddisV3()
    >>> downloader.download()

Note:
    To download data, instantiate the respective class from the `idownload` module and call the `download` method.

See Also:
    - `navigator.download.idownload`: The abstract base class for download interfaces.
"""

from .idownload.rinex.nasa_cddis import NasaCddisV3
from .idownload.sp3.ccdis_igs_sp3 import NasaCddisIgsSp3
