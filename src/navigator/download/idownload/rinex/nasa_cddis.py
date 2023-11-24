"""NASA CDDIS RINEX V3 Downloader Module.

This module facilitates the downloading of RINEX Version 3 (V3) files from the NASA CDDIS
(Crustal Dynamics Data Information System) FTP server. It offers a class, NasaCddis, that
enables establishing connections, checking the connection status, and downloading RINEX V3
files with specific criteria.

The module uses FTPFS from the fs.ftpfs package to interact with the FTP server and perform
file downloads. It leverages concurrent threading for efficient file retrieval.

Classes:
    NasaCddis: A class to download RINEX V3 files from the NASA CDDIS FTP server.

Attributes:
    None

Functions:
    None

Example:
    An example use case might be:

    ```
    downloader = NasaCddis(email="your_email@example.com", threads=8)
    downloader.connect()
    if downloader.is_alive():
        downloader.download(year=2023, day=300, save_path=Path("/path/to/save"))
    else:
        print("Connection to the FTP server is not available.")
    ```

Note:
    Ensure that the provided email address is valid and anonymous access is allowed on
    the NASA CDDIS FTP server.

Author:
    Nischal Bhattarai 
    nbhattarai@crimson.ua.edu

Version:
    0.1.0
"""
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from logging import NullHandler
from pathlib import Path

import tqdm
from fs.ftpfs import FTPFS

from ....utility.logger.logger import get_logger
from ....utility.matcher.matcher import GpsNav3DailyMatcher, MixedObs3DailyMatcher
from ..idownload import IDownload

__all__ = ['NasaCddisV3']


class NasaCddisV3(IDownload):
    """A class to download RINEX V3 files from NASA CDDIS FTP server.

    This class enables downloading RINEX V3 files from the NASA CDDIS (Crustal
    Dynamics Data Information System) FTP server. It provides methods to establish
    connections, check connection status, and download RINEX V3 files.

    Attributes:
        server_address (str): The address of the FTP server.

    Methods:
        __init__: Initializes the NasaCddis object.
        connect: Establishes a connection to the FTP server.
        is_alive: Checks if the FTP connection is alive.
        kwargs_checker: Validates the input arguments for the download method.
        download: Downloads RINEX V3 files from the FTP server.
    """

    server_address: str = "gdc.cddis.eosdis.nasa.gov"

    def __init__(self, email: str, threads: int = 5, logging: bool = False) -> None:
        """Initialize NasaCddis object to download RINEX V3 files.

        Args:
            email (str): Email address used for authentication.
            threads (int): Number of threads for concurrent downloads (default is 5).
            logging (bool): If True, enables logging (default is False).

        Raises:
            ValueError: If the provided email is invalid.
        """
        # Check if the email is valid
        if "@" not in email:
            raise ValueError(f"Invalid email address {email}")

        # Set a ftp filesystem
        self.email = email

        # Set the number of threads
        self.threads = threads

        # Get the logger
        self.logger = get_logger(__name__, dummy=not logging)

        # Disable logging if logging is False
        if not logging:
            self.logger.handlers.clear()
            self.logger.addHandler(NullHandler())

        # Matcher for GPS Nav Files
        self.gps_nav_matcher = GpsNav3DailyMatcher()
        self.obs_matcher = MixedObs3DailyMatcher()
        super().__init__(features="RinexV3 Download from NASA CDDIS")

    def connect(self) -> None:
        """Connects to the FTP server using FTPFS."""
        # Log the connection attempt
        self.logger.info(f"Connecting to {self.server_address}")
        self.ftpfs = FTPFS(
            host=self.server_address, user="anonymous", acct=self.email, tls=True
        )
        return

    def is_alive(self) -> bool:
        """Checks if the FTP connection is alive by performing a stat operation."""
        try:
            # Attempt to retrieve information about a specific path
            self.ftpfs.listdir("/")
            return True
        except Exception:
            return False

    def kwargs_checker(self, **kwargs) -> None:
        """Validates input arguments for the download method."""
        # Check if the year, day are in the kwargs
        if "year" not in kwargs or "day" not in kwargs or "save_path" not in kwargs:
            raise ValueError(
                f"year, day, save_path must be in kwargs for {self.__class__.__name__}"
            )

        self.logger.info(f"Year: {kwargs['year']}")
        self.logger.info(f"Day: {kwargs['day']}")
        self.logger.info(f"Save Path: {kwargs['save_path']}")

        # Day must be in range 1-366
        if kwargs["day"] < 1 or kwargs["day"] > 366:
            raise ValueError(
                f"day must be in range 1-366 for {self.__class__.__name__}"
            )

        # Year must be in range 1980-Now
        if kwargs["year"] < 1980 or kwargs["year"] > datetime.now().year:
            raise ValueError(
                f"year must be in range 1980-Now for {self.__class__.__name__}"
            )

        # Save Path must be a Path object
        if not isinstance(kwargs["save_path"], Path):
            raise ValueError(
                f"save_path must be a Path object for {self.__class__.__name__}"
            )

        # Check if the save_path exists
        if not kwargs["save_path"].exists():
            raise ValueError(f"save_path must exist for {self.__class__.__name__}")

        # CHeck if num_files is in kwargs
        if "num_files" in kwargs:
            self.logger.info(f"Number of Files to Download: {kwargs['num_files']}")
            if not isinstance(kwargs["num_files"], int):
                raise ValueError(
                    f"num_files must be an integer for {self.__class__.__name__}"
                )
            if not kwargs["num_files"] == -1 and not kwargs["num_files"] > 0:
                raise ValueError(
                    f"num_files must be greater than 0 or -1 (all files) for {self.__class__.__name__}"
                )

        return

    def ftp_download(self, ftp_file_path: str, save_path: Path) -> None:
        """Downloads a file from the FTP server while maintaining the same format.

        Args:
            ftp_file_path (str): The path to the file on the FTP server.
            save_path (Path): The path to save the file locally.

        Raises:
            ValueError: If the save_path is not a Path object.
        """
        # If not alive, reconnect
        if not self.is_alive():
            self.connect()

        # Get the file extension from the FTP file path
        file_extension = Path(ftp_file_path).suffix

        # Open the file in write binary mode and maintain the same file format
        with open(save_path.with_suffix(file_extension), "wb") as f:
            self.ftpfs.download(ftp_file_path, f)

        return

    def _download(self, *args, **kwargs) -> None:  # noqa : ARG006
        """Downloads RINEX V3 files from the FTP server.

        This method downloads RINEX V3 files for a specific day and year from
        the NASA CDDIS FTP server. It identifies matching observation and GPS
        navigation files, downloads them concurrently using multiple threads,
        and saves them to the specified local directory.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Required kwargs:
            year (int): The year to download RINEX V3 files.
            day (int): The day of the year to download RINEX V3 files.
            save_path (Path): The path to save the downloaded files.

        Optional kwargs:
            num_files (int): The number of files to download (default is -1).


        Raises:
            ValueError: If the provided input arguments are invalid.
        """
        self.logger.info(f"Downloading RINEX V3 files from {self.server_address}")

        # Check if the kwargs are valid
        self.kwargs_checker(**kwargs)

        # Get the year, day, save_path from the kwargs
        year = kwargs["year"]
        day = kwargs["day"]
        save_path = kwargs["save_path"]

        # Default path to daily data
        default_obs_path = (
            f"/pub/gnss/data/daily/{year}/{str(day).zfill(3)}/{str(year)[-2:]}d"
        )
        default_nav_path = (
            f"/pub/gnss/data/daily/{year}/{str(day).zfill(3)}/{str(year)[-2:]}n"
        )

        self.logger.debug(f"Default OBS Path: {default_obs_path}")
        self.logger.debug(f"Default NAV Path: {default_nav_path}")

        self.logger.info("Checking if the connection is alive!")
        # Get the file names under the default path
        if not self.is_alive():
            self.connect()

        # Get the file names under the default path
        obs_file_names = self.ftpfs.listdir(default_obs_path)
        nav_file_names = self.ftpfs.listdir(default_nav_path)

        # Get only the matched file names
        obs_file_names = [fname for fname in obs_file_names if self.obs_matcher(fname)]
        nav_file_names = [
            fname for fname in nav_file_names if self.gps_nav_matcher(fname)
        ]
        self.logger.info(f"Number of OBS Files: {len(obs_file_names)}")
        self.logger.info(f"Number of NAV Files: {len(nav_file_names)}")
        self.logger.debug(f"OBS File Names: {obs_file_names}")
        self.logger.debug(f"NAV File Names: {nav_file_names}")

        # Get the station name in both nav and obs
        obs_stations = [
            metadata["station_name"]
            for metadata in map(self.obs_matcher.extract_metadata, obs_file_names)
        ]
        nav_stations = [
            metadata["station_name"]
            for metadata in map(self.gps_nav_matcher.extract_metadata, nav_file_names)
        ]

        # Intersect the stations
        stations = set(obs_stations).intersection(set(nav_stations))
        self.logger.debug(f"Common Stations: {stations}")

        # File pairs
        file_pairs = []
        self.logger.info("Getting file pairs!")
        # Get (OBS, NAV) file per station
        for station_name in stations:
            station_pair = [None, None]
            for obs_fname in obs_file_names:
                if (
                    station_name
                    == self.obs_matcher.extract_metadata(obs_fname)["station_name"]
                ):
                    station_pair[0] = obs_fname
                    break
            for nav_fname in nav_file_names:
                if (
                    station_name
                    == self.gps_nav_matcher.extract_metadata(nav_fname)["station_name"]
                ):
                    station_pair[1] = nav_fname
                    break
            file_pairs.append(
                [
                    default_obs_path + "/" + station_pair[0],
                    default_nav_path + "/" + station_pair[1],
                ]
            )
        self.logger.info(f"Number of File Pairs: {len(file_pairs)}")
        # num_files in kwargs overrides the number of files to download
        if "num_files" in kwargs:
            if kwargs["num_files"] == -1:
                kwargs["num_files"] = len(file_pairs)  # Download all files

            self.logger.info(f"Number of Files to Download: {kwargs['num_files'] * 2}")
            num_files = kwargs["num_files"]
            if num_files > len(file_pairs):
                raise ValueError(
                    f"num_files must be less than or equal to {len(file_pairs)}"
                )
            file_pairs = random.sample(file_pairs, num_files)

        self.logger.info(f"Downloading Files with {self.threads} Threads!")
        with tqdm.tqdm(total=len(file_pairs) * 2, desc="Downloading") as pbar:
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                futures = []

                for obs_fname, nav_fname in file_pairs:
                    future_obs = executor.submit(
                        self.ftp_download, obs_fname, save_path / Path(obs_fname).name
                    )
                    future_nav = executor.submit(
                        self.ftp_download, nav_fname, save_path / Path(nav_fname).name
                    )
                    # Add callbacks to update the progress bar
                    future_obs.add_done_callback(
                        lambda x: pbar.update(1)  # noqa : ARG005
                    )
                    future_nav.add_done_callback(
                        lambda x: pbar.update(1)  # noqa : ARG005
                    )
                    futures.extend([future_obs, future_nav])

                # Wait for all the futures to complete
                executor.shutdown(wait=True)
        self.logger.info("Download Complete!")
        return
