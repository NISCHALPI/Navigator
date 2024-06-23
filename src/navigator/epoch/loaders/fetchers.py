"""Fetches the relevant data from the web.

This module provides functionality to fetch GPS navigation and SP3 data from the
NASA CDDIS repository. The data is downloaded for a specified date and station,
and then parsed to extract relevant information.

Classes:
    None

Functions:
    fetch_nav_data(date: datetime, logging: bool, station: str) -> tuple[pd.Series, pd.DataFrame]
    fetch_sp3(date: datetime, logging: bool, station: str) -> pd.DataFrame
"""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ...download.idownload.rinex.nasa_cddis import NasaCddisV3
from ...download.idownload.sp3.ccdis_igs_sp3 import NasaCddisIgsSp3
from ...parse.base_parse import Parser
from ...parse.iparse.nav.iparse_gps_nav import IParseGPSNav
from ...parse.iparse.sp3.iparse_sp3_gps import IParseSP3GPS

__all__ = ["fetch_nav_data", "fetch_sp3"]


# TODO: Add the compatibility for other GNSS systems
def fetch_nav_data(
    date: datetime,
    logging: bool = False,
    station: str = "JPL",
) -> tuple[pd.Series, pd.DataFrame]:
    """Fetches the navigation data from CDDIS for the given date.

    This function downloads the navigation data for a specified date and station
    from the CDDIS repository and parses it to extract navigation information.

    Args:
        date (datetime): The date for which the navigation data is to be fetched.
        logging (bool, optional): If True, logs will be printed to the console. Defaults to False.
        station (str, optional): The station from which the data is to be fetched. Defaults to "JPL".

    Returns:
        tuple[pd.Series, pd.DataFrame]: A tuple containing the navigation metadata as a pandas Series
                                         and the satellite positions as a pandas DataFrame.
    """
    # Fetch the data from CDDIS
    downloader = NasaCddisV3(logging=logging)
    parser = Parser(iparser=IParseGPSNav())

    # Set up a temporary directory
    with TemporaryDirectory() as temp_dir:
        # Download the navigation data
        downloader.download(
            year=date.year,
            day=date.timetuple().tm_yday,
            save_path=Path(temp_dir),
            num_files=1,
            match_string=station,  # Download from given station
        )

        # Get the navigation data file
        nav_file = list(Path(temp_dir).glob("*GN*"))[0]

        # Parse the navigation data
        nav_meta, nav_data = parser(nav_file)

    return nav_meta, nav_data


# TODO: Add the compatibility for other GNSS systems
def fetch_sp3(
    date: datetime,
    logging: bool = False,
) -> pd.DataFrame:
    """Fetches the SP3 data from CDDIS for the given date.

    This function downloads the SP3 data for a specified date and station
    from the CDDIS repository and parses it to extract satellite position information.

    Args:
        date (datetime): The date for which the SP3 data is to be fetched.
        logging (bool, optional): If True, logs will be printed to the console. Defaults to False.

    Returns:
        pd.DataFrame: The SP3 data as a pandas DataFrame.
    """
    # Fetch the data from CDDIS
    downloader = NasaCddisIgsSp3(logging=logging)
    parser = Parser(iparser=IParseSP3GPS())

    # Set up a temporary directory
    with TemporaryDirectory() as temp_dir:
        # Download the SP3 data
        downloader.download_from_datetime(
            time=date,
            save_dir=Path(temp_dir),
        )

        # Get the SP3 data file
        sp3_file = list(Path(temp_dir).glob("*SP3*"))[0]

        # Parse the SP3 data
        return parser(sp3_file)
