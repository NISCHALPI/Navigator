"""Download module to download EarthScope data from UNAVCO."""

# Link for API :https://gitlab.com/earthscope/public/earthscope-sdk
# TO DO: Add API to download data from UNAVCO
from ..idownload import IDownload

__all__ = ["IEarthScope"]


class IEarthScope(IDownload):
    """Class to download EarthScope data from UNAVCO."""

    pass
