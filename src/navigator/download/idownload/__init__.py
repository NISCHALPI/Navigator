"""Idownload interface module for Navigator.

Interface:
    - IDownload (abc.ABC): An abstract download interface.
    - NasaCddisV3 (IDownload): A concrete NASA CDDIS RINEX download interface.
    - NasaCddisIgsSp3 (IDownload): A concrete NASA CDDIS IGS SP3 download interface.

Example Usage:
    >>> from navigator.download import NasaCddisV3
    >>> downloader = NasaCddisV3()
    >>> downloader.download()

Note:
    To download data, instantiate the respective class from the `idownload` module and call the `download` method.

See Also:
    - `IDownload`: The abstract base class for download interfaces.
"""
