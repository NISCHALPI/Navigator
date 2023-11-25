"""This module provide Epoch Data Directory class which is ideal for ML/DL training.

WARNING: This module is only used for the directory structure that is already in Standard Format defined by the n
avigator.utility.v3daily.data.standerd_directory module. If you want to create a new directory structure, use the StanderdDirectory class.

The contents of the directory are organized into subdirectories by following this pattern: 

Pattern : {root}/{year}/{day_of_year}/{station}/(OBS|NAV)FRAG_{station}_{%Y%M%D}_{%H%M%S}.pkl

"""


import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterator

import tqdm

from ....parse.base_parse import Parser
from ....parse.iparse import IParseGPSNav, IParseGPSObs
from ...epoch.epoch import Epoch
from ...epoch.epochfragment import FragNav, FragObs
from ...igs_network import IGSNetwork
from ...logger.logger import get_logger
from ...matcher.fragment_matcher import FragNavMatcher, FragObsMatcher
from ...matcher.matcher import GpsNav3DailyMatcher, MixedObs3DailyMatcher
from .directory import AbstractDirectory
from .standerd_directory import StanderdDirectory

__all__ = ["EpochDirectory"]


class EpochDirectory(AbstractDirectory):
    """A directory containing epoch data."""

    def __init__(self, directory_path: str | Path, logging: bool = True) -> None:
        """Initializes a DailyEpochDirectory object.

        Args:
            directory_path (str | Path): The path to the directory where daily epoch data will be stored.
            triangulate (bool, optional): Whether to triangulate the epoch data. Defaults to True.
            logging (bool, optional): Whether to log the progress. Defaults to True.
        """
        # If doesn't exist, create the directory
        if not Path(directory_path).exists():
            Path(directory_path).mkdir(parents=True, exist_ok=True)

        # Set the logger
        self._logger = get_logger(__name__, not logging)

        # Set matcher for GPS NAV files
        self._gps_nav_matcher = GpsNav3DailyMatcher()
        self._obs_matcher = MixedObs3DailyMatcher()

        # Set matcher for fragments
        self._frag_nav_matcher = FragNavMatcher()
        self._frag_obs_matcher = FragObsMatcher()

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
        self._logger.info(f"Populating the directory with epoch files from {data_dir}.")
        # Clean the directory
        self.clean()
        self._logger.info(f"{self.directory_path} is cleaned.")

        # If the data directory is not a directory, raise an exception
        if not Path(data_dir).is_dir():
            raise Exception("The data directory is not a directory.")

        # If empty, raise an exception
        if len(list(Path(data_dir).iterdir())) == 0:
            raise Exception("The data directory is empty.")

        self._logger.info(f"Cleaning the data directory {data_dir}.")
        # Convert the data directory to a standard directory
        dir = StanderdDirectory(data_dir)

        # Clear the directory
        dir.clean()

        self._logger.info(f"Cloning the directory structure of {dir.directory_path}.")
        # Clone the directory structure of standard directory to the epoch directory
        EpochDirectory._clone_direcotry_structure(
            self.directory_path, data_dir, data_dir
        )

        # Total number of days directory
        total_days_dir = sum([len(list(year.iterdir())) for year in data_dir.iterdir()])
        self._logger.info(f"Total number of days directory: {total_days_dir}.")

        self._logger.info(f"Epochifying the data directory {data_dir}.")
        # Add Progress Bar
        with tqdm.tqdm(total=total_days_dir, desc="Epochifying Data Dir") as pbar:
            # Use process pool to process each day directory
            self._logger.info(
                f"Using {process} processes to epochify the data directory."
            )
            self._logger.disabled = True
            with ProcessPoolExecutor(max_workers=process) as executor:
                # Iterate over the year directories
                for year_dir in data_dir.iterdir():
                    for day_dir in year_dir.iterdir():
                        future = executor.submit(self._process_day, day_dir)

                        # Update the progress bar callback
                        future.add_done_callback(lambda p: pbar.update(1))  # noqa

                # Wait for all the processes to finish
                executor.shutdown(wait=True)
            self._logger.disabled = False

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

        # Log the station
        self._logger.info(f"Processing {station_dir.name}.")
        self._logger.info(f"OBS File: {obs_file.name}.")
        self._logger.info(f"NAV File: {nav_file.name}.")

        # Extract the metadata from the obs file
        file_met = self._obs_matcher.extract_metadata(obs_file.name)
        self._logger.info(f"OBS Metadata: {file_met}.")

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

        # Get the epoch fragments
        obs_fragments = FragObs.fragmentify(obs_data=obs_data, parent=obs_file.name)
        nav_fragments = FragNav.fragmentify(nav_data=nav_data, parent=nav_file.name)

        # Display the number of fragments
        self._logger.info(f"Number of OBS Fragments: {len(obs_fragments)}.")
        self._logger.info(f"Number of NAV Fragments: {len(nav_fragments)}.")

        self._logger.info(f"Dumping the fragments in {station_dir.name}.")
        # Dump the obs fragments in the station directory
        for obs_fragment in obs_fragments:
            if (
                len(obs_fragment.obs_data) >= 4
            ):  # Only dump the fragments with more than 4 satellites
                obs_fragment.save(
                    self.directory_path
                    / file_met["year"]
                    / file_met["day_of_year"]
                    / station_dir.name
                )

        self._logger.info(f"Dumping the fragments in {station_dir.name}.")
        # Dump the nav fragments in the station directory
        for nav_fragment in nav_fragments:
            if (
                len(nav_fragment.nav_data)
                >= 4  # Save the nav fragment if it has more than 4 satellites
            ):
                nav_fragment.save(
                    self.directory_path
                    / file_met["year"]
                    / file_met["day_of_year"]
                    / station_dir.name
                )

    def clean(self) -> None:
        """Deletes all the files in the directory."""
        # Clear the directory
        shutil.rmtree(self.directory_path)

        # Create the directory
        self.directory_path.mkdir(parents=True, exist_ok=True)
        return

    def _metadata_to_timefromgps(self, fname: str) -> float:
        """Converts the filename to timefromgps.

        Args:
            fname (str): The name of the file.

        Returns:
            float: The timefromgps.
        """
        matcher = (
            self._frag_obs_matcher
            if self._frag_obs_matcher.match(fname)
            else self._frag_nav_matcher
        )

        metadata = matcher.extract_metadata(fname)

        # GPS Time
        gps = datetime(1980, 1, 6, 0, 0, 0)

        year = int(metadata["year"])
        month = int(metadata["month"])
        day = int(metadata["day"])
        hour = int(metadata["hour"])
        minute = int(metadata["minute"])
        second = int(metadata["second"])

        return (datetime(year, month, day, hour, minute, second) - gps).total_seconds()

    def _previos_nav_file(self, index: int, all_file: list[Path]) -> Path:
        """Get the previous nav file from current index.

        Args:
            index (int): Index of the current obs fragment.
            all_file (_type_): list of all the files in the directory.

        Returns:
            Path: The previous nav fragment.
        """
        for i in range(index, -1, -1):
            if self._frag_nav_matcher.match(all_file[i].name):
                return all_file[i]
        return None

    def _next_nav_file(self, index: int, all_file: list[Path]) -> Path:
        """Get the next nav file from current index.

        Args:
            index (int): Index of the current obs fragment.
            all_file (_type_): list of all the files in the
            directory.

        Returns:
            Path: The next nav fragment.
        """
        for i in range(index, len(all_file)):
            if self._frag_nav_matcher.match(all_file[i].name):
                return all_file[i]
        return None

    def __iter__(self) -> Iterator["Epoch"]:
        """Iterate over the epoch files in the directory."""
        # Use glob to find all .epoch files in the directory
        all_files = self.directory_path.rglob("*.pkl")

        # Sort the file by timefromgps
        all_files = sorted(
            all_files, key=lambda x: self._metadata_to_timefromgps(x.name)
        )

        # Get the nearest nav file for each obs file
        for i, file in enumerate(all_files):
            # If the file is a obs file, yield the nav file
            if self._frag_obs_matcher.match(file.name):
                # Get the previous nav file
                prev_nav_file = self._previos_nav_file(i, all_files)

                # Get the next nav file
                next_nav_file = self._next_nav_file(i, all_files)

                # Get the nearest between the previous and next nav file
                nearest = None
                if prev_nav_file is None and next_nav_file is None:
                    raise Exception("No nav file found for the obs file.")
                if prev_nav_file is None:
                    nearest = next_nav_file
                elif next_nav_file is None:
                    nearest = prev_nav_file
                else:
                    prev_time = self._metadata_to_timefromgps(prev_nav_file.name)
                    next_time = self._metadata_to_timefromgps(next_nav_file.name)
                    curr_time = self._metadata_to_timefromgps(file.name)
                    nearest = (
                        prev_nav_file
                        if abs(curr_time - prev_time) < abs(curr_time - next_time)
                        else next_nav_file
                    )

                # Yield the epoch
                yield Epoch.load_from_fragment_path(
                    obs_frag_path=file, nav_frag_path=nearest
                )
