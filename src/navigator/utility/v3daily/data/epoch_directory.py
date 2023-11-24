"""This module provide Epoch Data Directory class which is ideal for ML/DL training.

WARNING: This module is only used for the directory structure that is already in Standard Format defined by the n
avigator.utility.v3daily.data.standerd_directory module. If you want to create a new directory structure, use the StanderdDirectory class.

The contents of the directory are organized into subdirectories by following this pattern: 

Pattern : {root}/{year}/{day_of_year}/{station}/EPOCH_{station}_{year}{day_of_year}{hour}{minute}.pkl

├── 2021
│   ├── 010
│   │   ├── ROAG00ESP
│   │   │   ├── ROAG00ESP_20210101000_01.pkl
│   │   │   ├── ROAG00ESP_20210101000_02.pkl
│   │   │   ├── ROAG00ESP_20210101000_03.pkl
│   │   │   ├── ROAG00ESP_20210101000_04.pkl
    ├── 011
│   │   ├── AMC000ARG
│   │   │   ├── AMC000ARG_20210101100_01.pkl
│   │   │   ├── AMC000ARG_20210101100_02.pkl
│   │   │   ├── AMC000ARG_20210101100_03.pkl
│   │   │   ├── AMC000ARG_20210101100_04.pkl
"""


import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import pandas as pd

from ....parse.base_parse import Parser
from ....parse.iparse import IParseGPSNav, IParseGPSObs
from ....satlib.triangulate import GPSIterativeTriangulationInterface, Triangulate
from ...epoch.epoch import Epoch
from ...igs_network import IGSNetwork
from ...matcher.matcher import GpsNav3DailyMatcher, MixedObs3DailyMatcher
from .directory import AbstractDirectory
from .standerd_directory import StanderdDirectory

__all__ = ["EpochDirectory"]


class EpochDirectory(AbstractDirectory):
    """A directory containing epoch data."""

    # Regex to match the epoch file. A matcher is provided in matcher utility module as well.
    epoch_regex = r"^EPOCH_([A-Z0-9]{8})_([0-9]{4})([0-9]{3})([0-9]{2})([0-9]{2})\.pkl$"

    def __init__(self, directory_path: str | Path, triangulate: bool = True) -> None:
        """Initializes a DailyEpochDirectory object.

        Args:
            directory_path (str | Path): The path to the directory where daily epoch data will be stored.
            triangulate (bool, optional): Whether to triangulate the epoch data. Defaults to True.
        """
        # If doesn't exist, create the directory
        if not Path(directory_path).exists():
            Path(directory_path).mkdir(parents=True, exist_ok=True)
        else:
            # If path exists, it must be an empty directory
            if len(list(Path(directory_path).iterdir())) != 0:
                raise ValueError(
                    "The directory is not empty. Clean the directory first or use a new directory."
                )

        # Set Trianguation Interface if triangulate is True
        self.traingulate = triangulate
        if triangulate:
            self._traingulator = Triangulate(GPSIterativeTriangulationInterface())

        # Set matcher for GPS NAV files
        self._gps_nav_matcher = GpsNav3DailyMatcher()
        self._obs_matcher = MixedObs3DailyMatcher()

        # Set the parser for GPS NAV files
        self._gps_nav_parser = Parser(IParseGPSNav())
        self._gps_obs_parser = Parser(IParseGPSObs())

        # Set the igs network
        self._igs_network = IGSNetwork()

        # Call the parent class constructor
        super().__init__(directory_path)

    @staticmethod
    def _clone_direcotry_structure(
        clone_root: Path, target_root: Path, curr_target_dir: Path
    ) -> None:
        """Clone the directory to the clone_root.

        Args:
            clone_root (Path): The root directory to clone the target directory.
            target_root (Path): The root directory of the target directory.
            curr_target_dir (Path): The current target directory to clone.
        """
        # Check if the current target directory is a directory
        if curr_target_dir.is_dir():
            # Check if the current target directory is not the target root directory
            if curr_target_dir != target_root:
                # Create the clone directory
                clone_dir = clone_root / curr_target_dir.relative_to(target_root)
                clone_dir.mkdir(parents=True, exist_ok=True)

            # Iterate over the files in the current target directory
            for files in curr_target_dir.iterdir():
                EpochDirectory._clone_direcotry_structure(
                    clone_root, target_root, files
                )
        # Do nothing if the current target directory is a file
        return

    def populate(self, data_dir: Path | str, process: int = 10) -> None:
        """Populate the directory with epoch files from the data directory.

        Args:
            data_dir (Path | str): The path to the data directory.
            process (int, optional): The number of processes to use. Defaults to 10.
        """
        # Clean the directory
        self.clean()

        # If the data directory is not a directory, raise an exception
        if not Path(data_dir).is_dir():
            raise Exception("The data directory is not a directory.")

        # If empty, raise an exception
        if len(list(Path(data_dir).iterdir())) == 0:
            raise Exception("The data directory is empty.")

        # Convert the data directory to a standard directory
        dir = StanderdDirectory(data_dir)

        # Clear the directory
        dir.clean()

        # Clone the directory structure of standard directory to the epoch directory
        EpochDirectory._clone_direcotry_structure(
            self.directory_path, dir.directory_path, dir.directory_path
        )

        with ProcessPoolExecutor(max_workers=process) as executor:
            # Iterate over the year directories
            for year_dir in data_dir.iterdir():
                for day_dir in year_dir.iterdir():
                    executor.submit(self._process_day, day_dir)

            # Wait for all the processes to finish
            executor.shutdown(wait=True)

        return

    def _process_day(self, day_dir: Path) -> None:
        """Populates the date directory with epoch files.

        Args:
            day_dir (Path): The path to the day of the Standard Directory.
        """
        # If there are more than two files in the directory, remove the directory
        for stations in day_dir.iterdir():
            # If there are more than two files in the directory, remove the directory
            # This is daily data so each station should have only two files (GPS NAV and MIXED OBS)
            if len(list(stations.iterdir())) != 2:
                shutil.rmtree(stations)
                return

            # Process the station
            self._process_station(stations)

        return

    def _process_station(self, station_dir: Path) -> None:
        """Process the station directory.

        Args:
            station_dir (Path): The path to the station directory.
        """
        # Get the obs and nav files in the station directory
        obs_file = [
            file for file in station_dir.iterdir() if self._obs_matcher.match(file.name)
        ][0]
        nav_file = [
            file
            for file in station_dir.iterdir()
            if self._gps_nav_matcher.match(file.name)
        ][0]

        # Extract the metadata from the obs file
        file_metadata = self._obs_matcher.extract_metadata(obs_file.name)

        # Parse the OBS and NAV using two threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Get the output of the parsers
            obs_future = executor.submit(self._gps_obs_parser.__call__, obs_file)

            nav_future = executor.submit(self._gps_nav_parser.__call__, nav_file)

            # Wait for both threads to finish
            executor.shutdown(wait=True)

        # Get the output of the parsers
        obs_metadata, obs_data = obs_future.result()
        nav_metadata, nav_data = nav_future.result()

        # If the obs data is empty, delete the station directory
        if len(obs_data) == 0 or len(nav_data) == 0:
            shutil.rmtree(station_dir)
            return

        # Epochify the data
        epoch_data = Epoch.epochify(obs=obs_data, nav=nav_data, mode="maxsv")

        # Iterate over the epochs
        for epoch in epoch_data:
            # Process the epoch
            self._process_epoch(
                epoch=epoch,
                obs_metadata=obs_metadata,
                nav_metadata=nav_metadata,
                year=file_metadata["year"],
                day=file_metadata["day_of_year"],
                station=station_dir.name,
            )

        pass

    def _process_epoch(
        self,
        epoch: Epoch,
        obs_metadata: pd.Series,
        nav_metadata: pd.Series,
        year: str,
        day: str,
        station: str,
    ) -> None:
        """Process the epoch for a given year, day and station.

        Args:
            epoch (Epoch): The epoch of data to process.
            obs_metadata (pd.Series): Metadata for the observations.
            nav_metadata (pd.Series): Metadata for the navigation data.
            year (str): The year of the epoch.
            day (str): The day of the year of the epoch.
            station (str): The station of the epoch.

        Returns:
            None
        """
        # If triangulation is enabled
        if self.traingulate:
            # Check if the epoch has at least 4 satellites
            if len(epoch) < 4:
                return
            try:
                # Triangulate the epoch and get the position of the receiver
                position = self._traingulator(
                    obs=epoch, obs_metadata=obs_metadata, nav_metadata=nav_metadata
                )
            except:  # noqa
                # If the epoch cannot be triangulated, return None
                return

            # Save the station id as the stationId attribute
            position["stationId"] = station

            try:
                # Set the station id
                position["realPosition"] = self._igs_network.get_xyz_from_matching_name(
                    station
                )
            except ValueError:
                # If the station id is not found, set the station id to None
                position["realPosition"] = None

            # Set the position series as the epoch attribute
            setattr(epoch, "position", position)

        # Save the epoch
        epoch.save(
            self.directory_path
            / year
            / day
            / station
            / f"EPOCH_{station}_{epoch.timestamp.strftime('%Y%m%d%H%M')}.pkl"
        )
        return

    def clean(self) -> None:
        """Deletes all the files in the directory."""
        # Clear the directory
        shutil.rmtree(self.directory_path)

        # Create the directory
        self.directory_path.mkdir(parents=True, exist_ok=True)
        return

    def __iter__(self) -> Iterator[Epoch]:
        """Iterate over the epoch files in the directory."""
        # Use glob to find all .epoch files in the directory
        epoch_files = self.directory_path.glob("**/*.epoch")

        # Use a generator expression to yield each epoch file
        return (Epoch.load(epoch_file) for epoch_file in epoch_files)
