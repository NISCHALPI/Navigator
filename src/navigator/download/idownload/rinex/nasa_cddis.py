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
from logging import NullHandler
from pathlib import Path

import tqdm

from ....logger.logger import get_logger
from ....utils.matcher.matcher import GpsNav3DailyMatcher, MixedObs3DailyMatcher
from ...ftpserver.ftpfs_server import FTPFSServer

__all__ = ["NasaCddisV3"]


class NasaCddisV3:
    """A class to download RINEX V3 files from NASA CDDIS FTP server.

    This class enables downloading RINEX V3 files from the NASA CDDIS (Crustal
    Dynamics Data Information System) FTP server. It provides methods to establish
    connections, check connection status, and download RINEX V3 files.

    Attributes:
        server_address (str): The address of the FTP server.
        username (str): The username for authentication.
        account (str): The account name for authentication.
        tls (bool): Indicates whether TLS encryption should be used.
    """

    server_address: str = "gdc.cddis.eosdis.nasa.gov"
    usename = "anonymous"
    tls = True

    def __init__(
        self,
        email: str = "anonymous@gmail.com",
        threads: int = 5,
        logging: bool = False,
    ) -> None:
        """Initialize NasaCddis object to download RINEX V3 files.

        Args:
            email (str): Email address used for authentication.
            threads (int): Number of threads for concurrent downloads (default is 5).
            logging (bool): If True, enables logging (default is False).

        Raises:
            ValueError: If the provided email is invalid.
        """
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

        self.logger.info(f"Instantiating NasaCddisV3 with email: {email}")
        # Initialize the FTPFS server
        self.ftpfs = FTPFSServer(
            host=self.server_address, user="anonymous", acct=email, tls=True
        )

        super().__init__()

    def _threaded_fetch_files(
        self,
        files: list[str],
        save_path: Path,
        no_pbar: bool = False,
        *args,  # noqa : ARG006
        **kwargs,  # noqa : ARG006
    ) -> None:
        """Fetches the file names from the FTP server.

        This method fetches the file names from the FTP server for the provided
        observation and navigation paths. It also updates the progress bar
        accordingly.

        Args:
            obs_paths (list[str]): The observation paths to fetch file names from.
            files (list[str]): The list to store the file names.
            save_path (Path): The path to save the downloaded files.
            no_pbar (bool): If True, disables the progress bar (default is False).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: If the provided input arguments are invalid.
        """
        # Log the number of files to download
        self.logger.info(f"Number of Files to Download: {len(files)}")
        # Initialize the progress bar
        with tqdm.tqdm(
            total=len(files), desc="Downloading", disable=not no_pbar
        ) as pbar:
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                futures = []

                for fname in files:
                    # Submit the download job to the executor
                    futures.append(
                        executor.submit(self.ftpfs.download, fname, save_path)
                    )
                    # Add a callback to update the progress bar
                    futures[-1].add_done_callback(
                        lambda x: pbar.update(1)  # noqa : ARG005
                    )

                # Wait for all the futures to complete
                executor.shutdown(wait=True)
        # Log the download completion
        self.logger.info("Download Complete!")
        self.logger.info(f"Downloaded {len(files)} files to {save_path.absolute()}")
        return

    def _search_available_files(
        self,
        year: int,
        day: int,
        match_string: str = None,
        sample: int = -1,
        *args,  # noqa : ARG006
        **kwargs,  # noqa : ARG006
    ) -> list[str]:
        """Searches for available files on the FTP server.

        This method searches for available files on the FTP server for the provided year, day, and match string.

        Args:
            year (int): The year of the RINEX files.
            day (int): The day of the RINEX files. [1-366]
            match_string (str): The string to match in the file names (default is None).
            sample (int): The number of files to sample (default is -1).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list[str]: The list of available files on the FTP server.
        """
        # Default path to daily data
        default_obs_path = (
            f"/pub/gnss/data/daily/{year}/{str(day).zfill(3)}/{str(year)[-2:]}d"
        )
        default_nav_path = (
            f"/pub/gnss/data/daily/{year}/{str(day).zfill(3)}/{str(year)[-2:]}n"
        )
        # Log the default paths
        self.logger.info(f"Default OBS Path: {default_obs_path}")
        self.logger.info(f"Default NAV Path: {default_nav_path}")

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

        # Filter stations based on match_string
        if match_string is not None:
            stations = [
                station for station in stations if match_string.upper() in station
            ]
            self.logger.info(f"Filtered Stations: {stations}")

        # Raise error if no stations are found
        if len(stations) == 0:
            raise ValueError("No stations found with the provided match_string")

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

        # Sample the file pairs if sample is not -1
        if sample != -1:
            # Sample must be less than or equal to the number of files
            if sample > len(file_pairs):
                raise ValueError(
                    f"requested sample must be less than or equal to {len(file_pairs)}"
                )
            file_pairs = random.sample(file_pairs, sample)
            # Log the sample
            self.logger.info(f"Number of File Pairs after sampling: {len(file_pairs)}")

        # Flatten the file pairs
        return [pair for pair in file_pairs for pair in pair]

    def download(
        self,
        year: int,
        day: int,
        save_path: Path,
        num_files: int = -1,
        no_pbar: bool = False,
        match_string: str = None,
        *args,  # noqa : ARG006
        **kwargs,  # noqa : ARG006
    ) -> None:
        """Downloads RINEX V3 files from the FTP server.

        This method downloads RINEX V3 files for a specific day and year from
        the NASA CDDIS FTP server. It identifies matching observation and GPS
        navigation files, downloads them concurrently using multiple threads,
        and saves them to the specified local directory.

        Args:
            year (int): The year of the RINEX files.
            day (int): The day of the RINEX files. [1-366]
            save_path (Path): The path to save the downloaded files.
            num_files (int): The number of files to download (default is -1).
            no_pbar (bool): If True, disables the progress bar (default is False).
            match_string (str): The string to match in the file names (default is None).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: If the provided input arguments are invalid.
        """
        # Check if the save path exists
        if not save_path.exists():
            raise ValueError("The save path does not exist.")

        # Get the available files
        available_files = self._search_available_files(
            year=year,
            day=day,
            match_string=match_string,
            sample=num_files,
            *args,
            **kwargs,
        )

        # Fetch the files
        self._threaded_fetch_files(
            available_files,
            save_path,
            no_pbar=no_pbar,
            *args,
            **kwargs,
        )

        # Close the FTPFS connection
        self.ftpfs.close()

        return
