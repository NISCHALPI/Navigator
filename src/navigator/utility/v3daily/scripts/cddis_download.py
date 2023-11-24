"""Download rinex files from CCIDS using curlftpfs."""
import calendar
import logging
import os
import random
import re
import shutil
import socket
import tempfile
from datetime import datetime
from pathlib import Path

import click
import tqdm

from ...ftpserver.mount_server import CDDISMountServer
from .logger import get_logger

# ------------------------------------------ Set the logging level ------------------------------

logger = get_logger(__name__)
logger.info(f"Staring the logging for {__name__}... ")

# ------------------------------------------ END ------------------------------

# ------------------------------------------ END ------------------------------

# ------------------------------------------ Global Variables ------------------------------
# USER EMAIL
USER_EMAIL = f"{os.getlogin()}@{socket.gethostname()}"

# Temporary directory name
TEMP_MOUNT_DIR_NAME = "tmp_ccids_mount"

# Regex to match the rinex files
_obs_regex = r".*_01D_30S_MO.*\.(crx|rnx)\.gz"
_nav_regex = r".*_01D*_GN.*\.(crx|rnx)\.gz"

# ------------------------------------------ END ------------------------------


# ------------------------------------------ Helper Functions ---------------------------------
# Helper function to match files in the given path and subdirectories
def _match_files(path: Path, regex: list[str]) -> list[str]:
    """Match files in the given path and subdirectories recursively."""
    matched_files = []
    for file in path.iterdir():
        if file.is_dir():
            matched_files.extend(_match_files(file, regex))
        elif file.is_file() and any(
            [re.match(pattern, file.name) for pattern in regex]
        ):
            logger.debug(f"Matched file: {file}")
            matched_files.append(file)
        else:
            logger.debug(f"Skipped file: {file}")
    return matched_files


# Format int 1 to 001
def format_int(num: int) -> str:
    """Format int to three digits. i.e 1 -> 001, 10 -> 010, 100 -> 100."""
    return f"{num:03}"


# Get the range of days from 1 to 366 for the given year and month
def get_day_range(year: int, month: int) -> tuple[int, int]:
    """Get the range of days from 1 to 366 for the given year and month.

    Args:
        year (int): The year.
        month (int): The month (1 to 12).

    Returns:
        tuple: A tuple containing the start and end day numbers.
    """
    start_date = datetime(year, month, 1)
    end_date = datetime(year, month, calendar.monthrange(year, month)[1])
    start_day = start_date.timetuple().tm_yday
    end_day = end_date.timetuple().tm_yday
    return start_day, end_day


# Daily Sweep Command
def _daily_sweep(
    mountDir: Path, save_path: Path, year: int, day: int, samples: int
) -> None:
    """Download RINEX files for the given year and day [1, 366]."""
    logger.info(f"Starting the daily sweep process for {year}/{day}...")

    # Make the save path if it does not exist
    logger.info(f"Making the save path if it does not exist: {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)

    # Point to the ftp path of the given year / month
    ftp_path = (
        mountDir / "pub" / "gnss" / "data" / "daily" / f"{year}" / format_int(day)
    )

    # YYD and YYN directories of the given year / month
    yyd = str(year)[-2:] + "d"
    yyn = str(year)[-2:] + "n"

    # Match the files at yyd
    logger.info(f"Matching the files at {ftp_path / yyd}")
    obs_files = _match_files(ftp_path / yyd, [_obs_regex])
    logger.info(f"Mached files at {ftp_path / yyd}: {obs_files.__len__()}")

    # Match the files at yyn
    logger.info(f"Matching the files at {ftp_path / yyn}")
    nav_files = _match_files(ftp_path / yyn, [_nav_regex])
    logger.info(f"Mached files at {ftp_path / yyn}: {nav_files.__len__()}")

    # Intersection of the stations in obs and nav files
    logger.debug("Finding the intersection of the stations in obs and nav files")
    nav_stations = set([file.name.split("_")[0] for file in nav_files])
    obs_stations = set([file.name.split("_")[0] for file in obs_files])

    # Intersection of the stations in obs and nav files
    logger.debug("Finding the intersection of the stations in obs and nav files")
    common_stations = nav_stations.intersection(obs_stations)
    logger.debug(f"Common stations: {common_stations}")

    # Filter paths with common stations
    logger.debug("Filtering paths with common stations")
    obs_files = [
        file for file in obs_files if file.name.split("_")[0] in common_stations
    ]
    nav_files = [
        file for file in nav_files if file.name.split("_")[0] in common_stations
    ]

    # Print the number of files
    logger.info(f"Number of Intersected Obs files: {obs_files.__len__()}")
    logger.info(f"Number of Intersected Nav files: {nav_files.__len__()}")

    # Check if the number of files are equal
    if obs_files.__len__() != nav_files.__len__():
        logger.info("Number of obs and nav files are not equal!")
        # Make one to one mapping between obs and nav files based on the station name
        logger.info("Get one obs, nav file for each station in common stations")
        stationMap = {}

        # Get one obs, nav file for each station in common stations
        for station in common_stations:
            stationMap[station] = [None, None]
            for obs_file in obs_files:
                if station == obs_file.name.split("_")[0]:
                    stationMap[station][0] = obs_file
                    break
            for nav_file in nav_files:
                if station == nav_file.name.split("_")[0]:
                    stationMap[station][1] = nav_file
                    break

        # Filter the files having both obs and nav files
        obs_file = [file[0] for file in stationMap.values() if file[0] is not None]
        nav_file = [file[1] for file in stationMap.values() if file[1] is not None]

        # Check if the number of files are equal
        logger.info(f"Update common number of files: {obs_file.__len__()}")

    # Sort and zip the nav and obs files (obs_path , nav_path)
    logger.debug("Sorting and zipping the nav and obs files")
    obs_files.sort(key=lambda x: x.name.split("_")[0])
    nav_files.sort(key=lambda x: x.name.split("_")[0])
    files = list(zip(obs_files, nav_files))

    # Take the samples if given
    if samples != -1 and samples < len(files):
        logger.debug(f"Taking {samples} random samples")
        files = random.sample(list(files), samples)

    # Copy the files to the destination path
    logger.info(f"Copying the  files to {save_path}")
    for obs_file, nav_file in tqdm.tqdm(
        files,
        desc="Copying files",
        total=len(files),
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    ):
        shutil.copy(obs_file, save_path / obs_file.name)
        shutil.copy(nav_file, save_path / nav_file.name)
        logger.debug(f"COPY: {obs_file} to {save_path / obs_file.name}")
        logger.debug(f"COPY: {nav_file} to {save_path / nav_file.name}")

    # Log the number of files copied
    logger.info(f"Copied {len(files)} files")
    return


# ------------------------------------------ END ------------------------------


# ------------------------------------------ Click Commands ------------------------------------------------
@click.group(invoke_without_command=True, no_args_is_help=True)
@click.pass_context
@click.option(
    "-e",
    "--email",
    required=False,
    default=USER_EMAIL,
    help=f"Email to login to CCIDS. Default: {USER_EMAIL}",
)
@click.version_option(version="1.0.0")
def main(ctx: click.Context, email: str) -> None:
    """Download RINEX files from CCIDS."""
    logger.info("Starting the download process...")

    # Set the context object to a dictionary
    ctx.ensure_object(dict)

    # Create a temporary directory to mount the ftp server
    mount_dir = Path(tempfile.gettempdir()) / TEMP_MOUNT_DIR_NAME
    logger.info(f"Creating a temporary mount directory: {mount_dir}")

    # Instantiate the CDDISMountServer
    logger.info("Instantiating the CDDISMountServer")
    mount_server = CDDISMountServer(mount_dir, email)

    # Add mount server to the context
    logger.info("Adding the mount server to the context")
    ctx.obj["mount_server"] = mount_server

    # Add end script callback to the context
    logger.debug("Adding the end script callback to the context")
    ctx.call_on_close(mount_server.unmount)

    return


@main.command()
@click.pass_context
@click.option(
    "-p",
    "--path",
    required=True,
    type=click.Path(exists=True, writable=True, path_type=Path),
    help="Path to save the files",
)
@click.option(
    "-y", "--year", required=True, type=click.INT, help="Year to download RINEX files"
)
@click.option(
    "-d",
    "--day",
    required=True,
    type=click.IntRange(1, 366),
    help="Day of year to download RINEX files",
)
@click.option(
    "-s",
    "--samples",
    required=False,
    type=click.INT,
    default=-1,
    help="Number of samples to download",
)
def daily(ctx: click.Context, path: Path, year: int, day: int, samples: int) -> None:
    """Download RINEX files for the given year and day."""
    # path / year / day
    path = path / str(year) / format_int(day)

    # Mount the server
    logger.info("Mounting the server")
    server: CDDISMountServer = ctx.obj["mount_server"]
    server.mount()
    # Sweep the given year and day
    _daily_sweep(server.mountDir, path, year, day, samples)
    return


@main.command()
@click.pass_context
@click.option(
    "-p",
    "--path",
    required=True,
    type=click.Path(exists=True, writable=True, path_type=Path),
    help="Path to save the files",
)
@click.option(
    "-y", "--year", required=True, type=click.INT, help="Year to download RINEX files"
)
@click.option(
    "-s",
    "--samples",
    required=False,
    type=click.INT,
    default=-1,
    help="Number of samples to download",
)
def yearly(ctx: click.Context, path: Path, year: int, samples: int) -> None:
    """Download RINEX files for the given year."""
    # path / year
    path = path / str(year)

    # Get the range of days
    start_day, end_day = 1, 366

    # If current year, get the range of days till yesterday
    if year == datetime.now().year:
        start_day = 1
        end_day = datetime.now().timetuple().tm_yday - 1

    # Mount the server
    logger.info("Mounting the server")
    server: CDDISMountServer = ctx.obj["mount_server"]
    server.mount()

    logger.info("Starting the yearly sweep process...")
    for day in range(start_day, end_day + 1):
        # path / year / day
        day_path = path / format_int(day)

        # Sweep the given year and day
        try:
            _daily_sweep(server.mountDir, day_path, year, day, samples)
            logger.info(
                f"------------------------Finished the daily sweep process for {year}/{day}----------------------------------"
            )
        except Exception as e:
            logger.error(f"Error in downloading files for {year}/{day}")
            logger.error(e)
            continue

    logger.info("Finished downloading all files")

    return


# ------------------------------------------ END ----------------------------------------------------------------------------------

# ------------------------------------------ Main Function ---------------------------------
if __name__ == "__main__":
    main()
# ------------------------------------------ END ---------------------------------
