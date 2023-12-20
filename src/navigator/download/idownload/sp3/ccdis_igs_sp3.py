"""Implementation of the NASA CDDIS SP3 download class."""
import os
from datetime import datetime, timedelta
from logging import NullHandler

from ....utility.ftpserver.ftpfs_server import FTPFSServer
from ....utility.logger.logger import get_logger
from ....utility.matcher.sp3_matcher import LegacySP3Matcher, SP3Matcher
from ...idownload import IDownload


class NasaCddisIgsSp3(IDownload):
    """Implementation of the NASA CDDIS SP3 download class.

    This class facilitates the download of SP3 files from the NASA CDDIS FTP server. It includes methods for handling
    GPS week/day conversions, initiating downloads, and managing FTP connections.

    Methods:
        __init__(self, email: str = "anonymous@gmail.com", logging: bool = False) -> None:
            Initializes the NasaCddisIgsSp3 instance with an optional email and logging configuration.

        gps_to_datetime(self, gps_week: int, gps_day: int) -> datetime:
            Converts GPS week and day to a datetime object.

        datetime_to_gps(self, dt: datetime) -> tuple[int, int]:
            Converts a datetime object to GPS week and day.

        _check_kwargs(self, kwargs: dict) -> None:
            Checks the validity of keyword arguments for download.

        _download(self, *args, **kwargs) -> None:
            Initiates the download of SP3 files from NASA CDDIS FTP server.

    """

    # Your module implementation here

    server_address: str = "gdc.cddis.eosdis.nasa.gov"
    usename = "anonymous"
    tls = True

    def __init__(
        self, email: str = "anonymous@gmail.com", logging: bool = False
    ) -> None:
        """Constructor for the NASA CDDIS SP3 download class.

        Args:
            email (str, optional): Email address to use for the download.
            logging (bool, optional): Indicates whether logging should be enabled. Defaults to False.

        Returns:
            None
        """
        # Set the email address
        self.email = email

        # Set the logger
        self.logger = get_logger(__name__)

        # Disable logging if requested
        if not logging:
            self.logger.handlers.clear()
            self.logger.addHandler(NullHandler())

        # Matcher for the SP3 files
        self.matcher = SP3Matcher()
        self.legacy_matcher = LegacySP3Matcher()

        # Set the FTP server
        self.logger.info(f"Setting the FTP server at {self.server_address}!")
        self.fs = FTPFSServer(self.server_address, self.usename, self.email, self.tls)

        super().__init__(features="NASA IGS SP3 Downloader")

    def gps_to_datetime(self, gps_week: int, gps_day: int) -> datetime:
        """This method is used to convert the GPS week and day to a datetime.

        Args:
            gps_week (int): GPS week number.
            gps_day (int): GPS day of week.

        Returns:
            datetime: Datetime object.
        """
        # GPS start date
        gps_start = datetime(1980, 1, 6)
        # Check the GPS week and day
        if gps_week < 0 or gps_day < 0:
            raise ValueError("GPS week or day cannot be negative!")

        return gps_start + timedelta(weeks=gps_week, days=gps_day)

    def datetime_to_gps(self, dt: datetime) -> tuple[int, int]:
        """This method is used to convert a datetime to GPS week and day.

        Args:
            dt (datetime): Datetime object.

        Returns:
            tuple: Tuple containing the GPS week and day.
        """
        # GPS start date
        gps_start = datetime(1980, 1, 6)
        # Check the datetime
        if dt < gps_start:
            raise ValueError("Datetime cannot be before GPS start date!")

        # Get the GPS week and day
        gps_week = (dt - gps_start).days // 7
        gps_day = (dt - gps_start).days % 7

        return gps_week, gps_day

    def _check_kwargs(self, kwargs: dict) -> None:
        """This method is used to check the keyword arguments.

        Args:
            kwargs (dict): Keyword arguments.

        Returns:
            None
        """
        # Check if the GPS week is provided
        if "gps_week" not in kwargs:
            raise ValueError("GPS week not provided!")

        # Check if the GPS day is provided
        if "gps_day" not in kwargs:
            raise ValueError("GPS day not provided!")

        # Check if the save directory is provided
        if "save_dir" not in kwargs:
            raise ValueError("Save directory not provided!")

        return

    def _download(self, *args, **kwargs) -> None:  # noqa: ARG002
        """This method is used to download the SP3 files from NASA CDDIS.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Required Keyword Args:
            gps_week (int): GPS week number.
            gps_day (int): GPS day of week.
            save_dir (str): Directory to save the downloaded files.

        Returns:
            None
        """
        self.logger.info("Downloading SP3 files from NASA CDDIS!")

        # Check the keyword arguments
        self._check_kwargs(kwargs)

        # Download the SP3 files
        self.dowload_from_week_info(
            kwargs["gps_week"], kwargs["gps_day"], kwargs["save_dir"]
        )

        return

    def fname(self, gps_week: int, gps_day: int) -> str:
        """This method is used to get the SP3 file name. Depending on the GPS week, the SP3 file name is generated.

        Args:
            gps_week (int): GPS week number.
            gps_day (int): GPS day of week.

        Returns:
            str: SP3 file name.
        """
        # Get the file to download
        fname = ""
        if gps_week <= 2237:
            fname = self.legacy_matcher.invert(gps_week=gps_week, day=gps_day)
        else:
            gps_d = self.gps_to_datetime(gps_week, gps_day)
            fname = self.matcher.invert(year=gps_d.year, day=gps_d.timetuple().tm_yday)

        return fname

    def dowload_from_week_info(
        self, gps_week: int, gps_day: int, save_dir: str
    ) -> None:
        """This method is used to download the SP3 files from NASA CDDIS.

        Args:
            gps_week (int): GPS week number.
            gps_day (int): GPS day of week.
            save_dir (str): Directory to save the downloaded files.

        Returns:
            None
        """
        prepath = f"/pub/gps/products/{gps_week}/"

        # Check if the file is available
        fname = os.path.join(prepath, self.fname(gps_week, gps_day))

        # Log the file to download
        self.logger.info(
            f"Downloading {fname}!: Is available? {self.fs.is_available(fname)}"
        )

        # Download the file
        self.fs.download(
            fname,
            save_path=save_dir,
        )

        return

    def download_from_datetime(self, time: datetime, save_dir: str) -> None:
        """This method is used to download the SP3 files from NASA CDDIS.

        Args:
            time (datetime): Datetime object.
            save_dir (str): Directory to save the downloaded files.

        Returns:
            None
        """
        # Get the GPS week and day
        gps_week, gps_day = self.datetime_to_gps(time)

        # Download the SP3 files
        self.dowload_from_week_info(gps_week, gps_day, save_dir=save_dir)
        return
