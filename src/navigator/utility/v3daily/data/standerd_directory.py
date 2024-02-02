"""Standard Directory Structure for RINEX v3 Daily Data."""

import shutil
from itertools import chain
from pathlib import Path
from typing import Iterator

from ...matcher.matcher import GpsNav3DailyMatcher, MixedObs3DailyMatcher
from .directory import AbstractDirectory

__all__ = ['StanderdDirectory']


class StanderdDirectory(AbstractDirectory):
    """Standard Directory Structure for RINEX v3 Daily Data.

    This module defines a standard directory structure for organizing RINEX v3 daily data files.
    The directory structure conforms to the naming convention outlined by the CDDIS for observation (OBS)
    and GPS navigation (NAV) files.

    Naming Convention:
        OBS Files: https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/daily_30second_data.html
        NAV (GPS) Files: https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/daily_gnss_n.html

    Directory Structure:
        {root}/{year}/{day_of_year}/{station_name}/{station_name}_{year}{day_of_year}{hour}{minute}_{frequency}.{file_extension}

    Example Directory Structure:
        .
        ├── 2021
        │   ├── 010
        │   │   ├── AMC200USA
        │   │   │   ├── AMC200USA_R_20210100000_01D_30S_MO.crx.gz
        │   │   │   └── AMC200USA_R_20210100000_01D_GN.rnx.gz
        │   │   ├── BAKO00USA
        │   │   │   ├── BAKO00USA_R_20210100000_01D_30S_MO.crx.gz
        │   │   │   └── BAKO00USA_R_20210100000_01D_GN.rnx.gz
        │   ├── 011
        │   │   ├── YELL00USA
        │   │   │   ├── YELL00USA_R_20210110000_01D_30S_MO.crx.gz
        │   │   │   └── YELL00USA_R_20210110000_01D_GN.rnx.gz
        │   │   ├── YELL01USA
        │   │   │   ├── YELL01USA_R_20210110000_01D_30S_MO.crx.gz
        │   │   │   └── YELL01USA_R_20210110000_01D_GN.rnx.gz
        │   └── 014
        │       ├── CEDR00USA
        │       │   ├── CEDR00USA_R_20210140000_01D_30S_MO.crx.gz
        │       │   └── CEDR00USA_R_20210140000_01D_GN.rnx.gz
        │       ├── CEDR01USA
        │       │   ├── CEDR01USA_R_20210140000_01D_30S_MO.crx.gz
        │       │   └── CEDR01USA_R_20210140000_01D_GN.rnx.gz

    This directory structure is used for both OBS and NAV files collected from any source.

    Attributes:
        directory_path (str or Path): Path to the root directory.

    Methods:
        __init__: Initializes the StanderdDirectory object.
        clean: Cleans the directory structure.
        __iter__: Iterates over the files in the directory.
    """

    def __init__(self, directory_path: str | Path, clean: bool = False) -> None:
        """Initialize the StanderdDirectory object.

        Args:
            directory_path (str or Path): Path to the root directory.
            clean (bool): If True, preprocesses the directory to adhere to the standard structure.
        """
        super().__init__(directory_path)

        # Matcher for GPS Files
        self.gps_nav_matcher = GpsNav3DailyMatcher()
        self.mixed_obs_matcher = MixedObs3DailyMatcher()

        # Preprocess the directory if clean is True
        if clean:
            self.clean()

    @staticmethod
    def _flatten(path: Path, root_path: Path, root: bool = True) -> None:
        """Recursively flatten the directory structure so that all the files are in the root directory and deletes the empty directories."""
        if path.is_dir():
            for child in path.iterdir():
                StanderdDirectory._flatten(child, root_path, False)

            if not root:
                path.rmdir()

        elif path.is_file():
            if not root:
                # Move the file to the root directory
                path.replace(root_path / path.name)
        return

    def clean(self) -> None:
        """Cleans the directory structure to adhere to the standard structure."""
        # Flatten the directory structure
        self._flatten(self.directory_path, self.directory_path)

        # Obs files metdata
        obs_metadata = []
        nav_metadata = []

        # Remove anything that does not match either the GPS NAV or MIXED OBS
        for file in self.directory_path.iterdir():
            if not self.gps_nav_matcher.match(
                file.name
            ) and not self.mixed_obs_matcher.match(file.name):
                # Remove the file
                file.unlink()

            elif self.gps_nav_matcher.match(file.name):
                # Extract the metadata
                nav_metadata.append(self.gps_nav_matcher.extract_metadata(file.name))
                nav_metadata[-1]["file_path"] = file

            elif self.mixed_obs_matcher.match(file.name):
                # Extract the metadata
                obs_metadata.append(self.mixed_obs_matcher.extract_metadata(file.name))
                obs_metadata[-1]["file_path"] = file

        # Union the metadata
        metadata = obs_metadata + nav_metadata

        # Get the set of unique years
        years = set([file["year"] for file in metadata])

        # Create the year directories
        for year in years:
            (self.directory_path / year).mkdir(exist_ok=True)

        # Create the day directories for each year
        for year in years:
            days = set(
                [file["day_of_year"] for file in metadata if file["year"] == year]
            )
            for day in days:
                (self.directory_path / year / day).mkdir(exist_ok=True)

        # Move the files to appropriate directories
        for file in metadata:
            # Get the file path
            file_path = file["file_path"]

            # Get the year and day of year
            year = file["year"]
            day_of_year = file["day_of_year"]

            # Get the station name
            station_name = (
                file["marker_name"]
                + file["marker_number"]
                + file["receiver_number"]
                + file["country_code"]
            )

            # Create the station directory if it does not exist
            (self.directory_path / year / day_of_year / station_name).mkdir(
                exist_ok=True
            )
            # Move the file to the appropriate directory
            file_path.replace(
                self.directory_path / year / day_of_year / station_name / file_path.name
            )

        # Remove invalid directories
        self.remove_invalid_dirs(self.directory_path)

        return

    def _is_leaf(self, path: Path) -> bool:
        """Returns True if the path is a leaf directory i.e dirctory containing files only."""
        if any([child.is_file() for child in path.iterdir()]):
            return True
        return False

    def remove_invalid_dirs(self, root: Path) -> None:
        """Recursively removes invalid directories based on specific criteria.

        Criteria:
            1. If the directory is empty.
            2. If the leaf directory does not contain one nav file and one obs file.

        Args:
            root (Path): The root directory to start the removal process.

        Returns:
            None
        """
        # If the directory is empty
        if len(list(root.iterdir())) == 0:
            # Remove the directory
            root.rmdir()
            return

        if not self._is_leaf(root):
            # Recurse on the children
            for child in root.iterdir():
                self.remove_invalid_dirs(child)

            # Recheck if the directory is empty
            if len(list(root.iterdir())) == 0:
                # Remove the directory
                root.rmdir()
                return
        else:
            # Check that the leaf directory contains one nav file and one obs file
            nav_files = [
                child
                for child in root.iterdir()
                if self.gps_nav_matcher.match(child.name)
            ]
            obs_files = [
                child
                for child in root.iterdir()
                if self.mixed_obs_matcher.match(child.name)
            ]

            if len(nav_files) != 1 or len(obs_files) != 1:
                # Remove the directory
                shutil.rmtree(root)
                return

    def __iter__(self) -> Iterator[Path]:
        """Iterate over the files in the directory."""
        # Return the rinex files in the directory i.e (file with .rnx.gz or .crx.gz extension)
        # Glob pattern: *.rnx.gz or *.crx.gz
        return iter(
            chain(
                self.directory_path.glob("*.rnx.gz"),
                self.directory_path.glob("*.crx.gz"),
            )
        )
