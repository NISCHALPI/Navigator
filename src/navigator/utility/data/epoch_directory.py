"""This module provide Epoch Data Directory class which is ideal for ML/DL training.

Each RINEX obs file is broken down into observation epochs and each epoch is stored in a separate file. The directory tree is as follows:

    Data/
    ├── 00NA
    │   ├── 20220101
    │   │   ├── EPOCH_00NA_20220101_000000.obs

"""


import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from ...parse import IParseGPSNav, IParseGPSObs, Parser
from ...satlib.triangulate import GPSIterativeTriangulationInterface, Triangulate
from ..epoch.epoch import Epoch
from ..igs.igs_network import IGSNetwork
from .directory import AbstractDirectory
from .igs_daily_data_directory import IgsDailyDataDirectory

__all__ = ["DailyEpochDirectory"]



class DailyEpochDirectory(AbstractDirectory):
    """This class represents a Daily Epoch Data Directory, ideal for ML/DL training. Each RINEX obs file is broken down into observation epochs, and each epoch is stored in a separate file.
    
    The directory structure follows this pattern:

    Data/
    ├── 00NA
    │   ├── 20220101
    │   │   ├── EPOCH_00NA_20220101_000000.epoch

    Args:
        directory_path (str | Path): The path to the directory where daily epoch data will be stored.

    Attributes:
        _obs_regex (re.Pattern): Regular expression for matching RINEX observation files.

    Methods:
        __init__(self, directory_path: str | Path) -> None:
            Initializes a DailyEpochDirectory object.

        populate(self, igs_directory: IgsDailyDataDirectory, threads: int = 10) -> None:
            Populates the epoch data directory with data from the IGS data directory.

        _populate_date_directory(self, date_dir: Path) -> None:
            Populates the date directory with epoch files.

    Example:
        # Create a DailyEpochDirectory instance
        epoch_directory = DailyEpochDirectory("Data/00NA/20220101")

        # Populate the epoch directory from an IGS data directory
        igs_directory = IgsDailyDataDirectory("IGSData/00NA/20220101")
        epoch_directory.populate(igs_directory)
    """
    _obs_regex = re.compile(r".*_(R|S|U)_.*_(MO|GO)\.(crx|rnx)\.gz$")
    _nav_regex = re.compile(r".*_(R|S|U)_.*_(MN|GN)\.(crx|rnx)\.gz$")
    
    def __init__(self, directory_path: str | Path) -> None:
        """Initializes a DailyEpochDirectory object.

        Args:
            directory_path (str | Path): The path to the directory where daily epoch data will be stored.
        """
        # If doesn't exist, create the directory
        if not Path(directory_path).exists():
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            
        # Set Trianguation Interface
        self._traingulator = Triangulate(GPSIterativeTriangulationInterface())

        # Set the igs network
        self._igs_network = IGSNetwork()
        
        # Call the parent class constructor        
        super().__init__(directory_path)
        
        
    def _copy_igs_tree(self, igs_directory: IgsDailyDataDirectory) -> None:
            """Copies the contents of an IGS daily data directory to the epoch directory.

            Args:
                igs_directory (IgsDailyDataDirectory): The IGS daily data directory to copy from.
                    Must be paired with a corresponding IGS daily data directory.

            Raises:
                Exception: If the IGS directory is not paired.
            """
             # Check if the IGS directory is paired, if not raise an exception
            if not igs_directory.is_paired():
                raise Exception("The IGS directory is not paired. Please pair it first.")

            # Copy the everything inside the IGS directory to the epoch directory
            shutil.copytree(igs_directory.directory_path, self.directory_path, dirs_exist_ok=True)
            
            return
        
    
    def populate(self, igs_directory: IgsDailyDataDirectory, process : int = 10) -> None:
        """Populates the epoch data directory with data from the IGS data directory.

        Args:
            igs_directory (IgsDataDirectory): The IGS data directory to copy data from.
            process (int, optional): The number of processes to use. Defaults to 10.
        """
        # Copy the IGS directory to the epoch directory
        self._copy_igs_tree(igs_directory)
        
        # Thread pool for parallel processing
        process_pool = ProcessPoolExecutor(max_workers=process)
        
        
        # Iterate over all the files in the directory
        for path_to_station in self.directory_path.iterdir():
            # If it is a directory
            if path_to_station.is_dir():
                # Iterate over all the date in the directory
                for date_dir in path_to_station.iterdir():
                    # Submit the task to the thread pool
                    process_pool.submit(self._populate_date_directory, date_dir)    
                
        # Shutdown the thread pool after all the tasks are completed
        process_pool.shutdown(wait=True)
        
        return
        
        
    def _populate_date_directory(self, date_dir : Path) -> None:
        """Populates the date directory with epoch files.

        Args:
            date_dir (Path): The path to the date directory.
        """
        # If there are more than two files in the directory, remove the directory
        if len(list(date_dir.iterdir())) > 2:
            shutil.rmtree(date_dir)
            return
        
        # Get the rinex obs file in the directory by matching the regex
        for files in date_dir.iterdir():
            if self._obs_regex.match(files.name):
                obs_file = files
                break
        
        # Get the rinex nav file in the directory by matching the regex
        for files in date_dir.iterdir():
            if self._nav_regex.match(files.name):
                nav_file = files
                break
            
        
        
        # If there is no obs file or nav file, remove the directory
        if not obs_file or not nav_file:
            shutil.rmtree(date_dir)
            return
        
        
        # Parse the obs file
        obs_parser = Parser(iparser=IParseGPSObs())
        
        # Get the metadata and data from the obs file as pd.Series, pd.DataFrame
        metadata , obs_data = obs_parser(obs_file)
        
        # If cannot parse the obs file, remove the directory
        if len(obs_data) == 0:
            shutil.rmtree(date_dir)
            return
        
        # Parse the nav file
        nav_parser = Parser(iparser=IParseGPSNav())
        
        # Get the metadata and data from the nav file as pd.Series, pd.DataFrame
        nav_metadata , nav_data = nav_parser(nav_file)
        
        # If cannot parse the nav file, remove the directory
        if len(nav_data) == 0:
            shutil.rmtree(date_dir)
            return
    
        # Epochify the obs file
        epoches = Epoch.epochify(obs_data)
        
            
        # Process each epoch
        for epoch in epoches:
            self._process_epoch(
                epoch=epoch,
                obs_metadata=metadata,
                nav_file=nav_data,
                nav_metadata=nav_metadata,
                save_dir=date_dir
            )
            
        # Remove the obs file
        obs_file.unlink()
        # Remove the nav file
        nav_file.unlink()
        
        return
    
    def _process_epoch(self, epoch : Epoch, obs_metadata : pd.Series, nav_file : pd.DataFrame , nav_metadata : pd.Series, save_dir: Path) -> None:
            """Process a single epoch of data by triangulating the observations and saving the resulting position.

            Args:
                epoch (Epoch): The epoch of data to process.
                obs_metadata (pd.Series): Metadata for the observations.
                nav_file (pd.DataFrame): The navigation data.
                nav_metadata (pd.Series): Metadata for the navigation data.
                save_dir (Path): The path to the directory where the epoch file will be saved.

            Returns:
                None
            """
            # Check if the epoch has at least 4 satellites
            if len(epoch) < 4:
                return
            
            try:
                # Triangulate the epoch and get the position of the receiver
                position = self._traingulator(
                    obs=epoch,
                    obs_metadata=obs_metadata,
                    nav=nav_file,
                    nav_metadata=nav_metadata
                )
            except: # noqa
                # If the epoch cannot be triangulated, return None
                return
            
            # Get the marker name from the metadata
            marker_name : str = obs_metadata["MARKER NAME"]
            # Process the marker name to get the station id
            marker_name = marker_name.strip()
            
            # Save the station id as the stationId attribute
            position["stationId"] = marker_name
            
            try:
                # Set the station id
                position["realPosition"] = self._igs_network.get_xyz(marker_name)
            except ValueError:
                # If the station id is not found, set the station id to None
                position["realPosition"] = None
            
            # Set the position series as the epoch attribute
            setattr(epoch, "position", position)
            
            # Save the epoch
            epoch.save(save_dir / f"{marker_name}_EPOCH_{epoch.timestamp.strftime('%Y%m%d_%H%M%S')}.epoch")
            
            return
    
    def clean(self) -> None:
        """Override the clean method of the parent class to remove empty directories."""
        pass        
        
    
    

        
        
    
    
    