"""This module contains the IgsDataDirectory class.

Definition:
- Data Directory: A directory that has IGS station code as sub-directories, and in each sub-directory, there are 'MO' and 'MN' Rinex files. For example:

```bash
    tree ./Data/. 

    Data/
    ├── 00NA
    ├── ABMF
    │   ├── ABMF00GLP_R_20213110000_01D_30S_MO.crx.gz
    │   ├── ABMF00GLP_R_20213120000_01D_30S_MO.crx.gz
    │   ├── ABMF00GLP_R_20220300000_01D_30S_MO.crx.gz
    │   ├── ABMF00GLP_R_20220310000_01D_30S_MO.crx.gz
    │   ├── ABMF00GLP_R_20220380000_01D_30S_MO.crx.gz
    │   └── ABMF00GLP_R_20220390000_01D_30S_MO.crx.gz
    ├── ABNY
"""
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from warnings import warn

from ...download.idownload.rinex.aus_gov import AusGovDownload
from .directory import AbstractDirectory

__all__ = ["IgsDailyDataDirectory"]


class IgsDailyDataDirectory(AbstractDirectory):
    """A class representing a directory containing IGS data files.

    Attributes:
        _gps_file_regex (re.Pattern): A regular expression to match the RINEX file name.
        _date_regex (re.Pattern): A regular expression to match the date in the RINEX file name.
        _obs_regex (re.Pattern): A regular expression to match the observation file name.
        _nav_regex (re.Pattern): A regular expression to match the navigation file name.
        

    Methods:
        clean: Cleans the data directory by removing the files that are not RINEX files under the station directories. Removes the empty station directories as well.
        pair: Pairs each observation file with its corresponding navigation file for that day.
        

    Args:
        directory_path (str | Path): The path to the directory containing IGS data files.
        clean (bool, optional): If True, the directory will be cleaned by removing non-RINEX files and empty station directories. Defaults to False.
    """

    # Regular expression to match the RINEX file name
    _gps_file_regex = re.compile(r".*_(R|S|U)_(\d{11})_0.*\.(crx|rnx)\.gz$")  # Eg: ABMF00GLP_R_20213110000_01D_30S_MO.crx.gz
    _date_regex = re.compile(r".*_(\d{4})(\d{3})(\d{2})(\d{2})_.*")  # Eg: ABMF00GLP_R_20213110000_01D_30S_MO.crx.gz(_YYYYDDDHHMM_)
    _obs_regex = re.compile(r".*_(R|S|U)_.*_(MO|GO)\.(crx|rnx)\.gz$")  # Eg: ABMF00GLP_R_20213110000_01D_30S_MO.crx.gz
    _nav_regex = re.compile(r".*_(R|S|U)_.*_(MN|GN)\.(crx|rnx)\.gz$")  # Eg: ABMF00GLP_R_20213110000_01D_30S_MN.crx.gz

    def __init__(self, directory_path: str | Path, clean: bool = False, pair : bool = False) -> None:
        """Constructor of IgsDataDirectory.

        Args:
            directory_path (str | Path): The path to the directory containing IGS data files.
            clean (bool, optional): If True, the directory will be cleaned by removing non-RINEX files and empty station directories. Defaults to False.
            pair (bool, optional): If True, the directory will be paired by pairing each observation file with its corresponding navigation file for that day. Defaults to False.
        """
        super().__init__(directory_path)

        # Clean the directory if clean is True
        if clean:
            self.clean(dryrun=False)

        # Pair the directory if pair is True
        if pair:
            self.pair(dryrun=False)
        
        # Add the downloader
        self._downloader = AusGovDownload(max_workers=10)
        

    def clean(self, dryrun: bool = True) -> None:
        """Cleans the data directory by removing the files that are not RINEX files under the station directories. Removes the empty station directories as well.

        Args:
            dryrun (bool, optional): If True, the method will not actually remove any files or directories. Defaults to False.
        """
        rmfilecount = 0
        rmdircount = 0
        # If already paired, raise an error
        if self.is_paired():
            raise RuntimeError(f"{self.directory_path} is already paired. Do not clean paired directories. Data Loss may occur.")
        
        # Iterate over the station directories
        for station in self:
            # Iterate over the files in the station directory
            for file in station.iterdir():
                # Check if the file is a RINEX file
                if not self._gps_file_regex.match(file.name):
                    # Remove the file
                    if not dryrun:
                        os.remove(file)
                    print(f"Removed {file}.")
                    rmfilecount += 1

            # Check if the station directory is empty
            if not list(station.iterdir()):
                # Remove the station directory
                print(f"Removed {station}.")
                if not dryrun:
                    os.rmdir(station)
                rmdircount += 1

        # Print the number of files and directories removed
        print(f"Removed {rmfilecount} files and {rmdircount} directories.")
        return

    def stamp_directory_after_pairing(self) -> None:
        """Stamps the directory after pairing by adding a '.paired' file to the directory with the date and time of pairing."""
        with open(self.directory_path / ".paired", "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        return    
    
    def is_paired(self) -> bool:
        """Checks if the directory is paired by checking if it has a '.paired' file.

        Returns:
            bool: True if the directory is paired, False otherwise.
        """
        return (self.directory_path / ".paired").exists()
    
    def pair(self, dryrun: bool = True) -> None:
        """Pairs each observation file with its corresponding navigation file for that day.

        Args:
            dryrun (bool, optional): If True, the method will not actually remove any files or directories. Defaults to False.
        """
        # If the directory is already paired, raise a warning and return
        if self.is_paired():
            warn(f"{self.directory_path} is already paired. Remove the '.paired' file to pair again.")
            return
        
        # Iterate over the station directories
        for station in self:
            obs_files_by_date = {}
            nav_files_by_date = {}
            # Iterate over the files in the station directory and extrate dates and obs files for each date
            for file in station.iterdir():
                # If not a file, continue
                if not file.is_file():
                    continue
                
                # If the file is an RIENX observation file, extract the date
                if self._obs_regex.match(file.name):
                    # Extract the date of the file and make it _YYYYDDDHHMM_
                    date = self._date_regex.match(file.name).groups()
                    date = f"_{date[0]}{date[1]}{date[2]}{date[3]}_"
                    
                   
                    # Add the file to the list of obs files for that date
                    if date in obs_files_by_date:
                        obs_files_by_date[date].append(file)
                    else:
                        obs_files_by_date[date] = [file]
                        
                # If the file is an RIENX navigation file, extract the date
                elif self._nav_regex.match(file.name):
                    # Extract the date of the file and make it _YYYYDDDHHMM_
                    date = self._date_regex.match(file.name).groups()
                    date = f"_{date[0]}{date[1]}{date[2]}{date[3]}_"
                    
                    # Add the file to the list of nav files for that date
                    if date in nav_files_by_date:
                        nav_files_by_date[date].append(file)
                    else:
                        nav_files_by_date[date] = [file]
                        
            # Intersect the dates of obs and nav files
            dates = set(obs_files_by_date.keys()).intersection(set(nav_files_by_date.keys()))
            
            # If there is no intersection, remove the station directory, warn the user and continue
            if dates.__len__() == 0:
                print(f"No Common Dates : Removing {station}.")
                if not dryrun:
                    shutil.rmtree(station)
                continue
            
            # If there is an intersection, make the date dir and copy crossponding obs and nav files to it
            for date in dates:
                # Make a date directory
                datedir = station / date.replace("_", "")
                if not datedir.exists():
                    print(f"Making {datedir}.")
                    if not dryrun:
                        datedir.mkdir()
                        
                # Copy the obs files to the date directory
                for obs_file in obs_files_by_date[date]:
                    print(f"Copying {obs_file} to {datedir}.")
                    if not dryrun:
                        shutil.copy(obs_file, datedir)
                        
                # Copy the nav files to the date directory
                for nav_file in nav_files_by_date[date]:
                    print(f"Copying {nav_file} to {datedir}.")
                    if not dryrun:
                        shutil.copy(nav_file, datedir)
            
            
            # Remove the redundant files under the station directory if they are not copied to the date directories
            for file in station.iterdir():
                if file.is_file():
                    print(f"Removing {file}.")
                    if not dryrun:
                        os.remove(file)
        
        # Stamp the directory after pairing
        self.stamp_directory_after_pairing()
        return

        