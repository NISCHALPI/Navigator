"""Load the rinex data to an epoch object."""

import typing as tp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ...epoch.epoch import Epoch
from ...parse.iparse.nav.iparse_gps_nav import IParseGPSNav
from ...parse.iparse.obs.iparse_gps_obs import IParseGPSObs
from .fetchers import fetch_sp3

__all__ = [
    "get_fragments_idx_from_nav",
    "get_sp3_data",
    "get_noon_of_unique_days",
    "from_rinex_dataframes",
    "from_precise_ephemeris",
]

  
def get_sp3_data(
    timestamps: tp.List[pd.Timestamp],
    max_workers: int = 4,
    logging: bool = False,
) -> pd.DataFrame:
    """Get the SP3 data for the timestamps.

    Args:
        timestamps (List[pd.Timestamp]): A list of timestamps.
        max_workers (int, optional): The maximum number of workers. Defaults to 4.
        logging (bool, optional): If True, the logging will be enabled. Defaults to False.

    Returns:
        pd.DataFrame: The SP3 data.

    Raises:
        ValueError: If the SP3 data is not available.
    """
    # Get the noon of the unique days
    unique_days = get_noon_of_unique_days(timestamps)

    # Thread the fetching of the SP3 data
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        sp3_data = executor.map(fetch_sp3, unique_days, [logging] * len(unique_days))

    return pd.concat([sp3[1] for sp3 in sp3_data], axis=0).sort_index()

def get_noon_of_unique_days(timestamps: tp.List[pd.Timestamp]) -> tp.List[pd.Timestamp]:
    """Get the noon of the unique days in the timestamps.

    Args:
        timestamps (List[pd.Timestamp]): A list of timestamps.

    Returns:
        List[pd.Timestamp]: A list of timestamps at noon of the unique days in the timestamps.
    """
    # Normalize timestamps to get unique days
    unique_days = set([timestamp.normalize() for timestamp in timestamps])

    # Add the noon timestamp of the unique days
    delta = pd.Timedelta(hours=12)

    return [day + delta for day in unique_days]

# TODO: Add the compatibility for other GNSS systems
def get_fragments_idx_from_nav(
    observation_data: pd.DataFrame, 
    navigation_data: pd.DataFrame,
    mode: str = "maxsv",
    matching_threshold: pd.Timedelta = pd.Timedelta(hours=3),
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Matches the observation timestamps to the navigation timestamps based on the mode.
    
    The dataframes must have a MultiIndex with the levels 'time' and 'sv'.
    
    Args:
        observation_data (pd.DataFrame): Observation data.
        navigation_data (pd.DataFrame): Navigation data.
        mode (str, optional): Mode to match the timestamps (maxsv | nearest). Defaults to "maxsv".
        matching_threshold (pd.Timedelta, optional): Matching threshold within which the timestamps are matched. Defaults to pd.Timedelta(hours=3).
    
    Returns:
        list[tuple[pd.Timestamp, pd.Timestamp]]: List of matched timestamps.
    """
    # Get the timestamps of the observations.
    obsTimestamps = observation_data.index.get_level_values("time").unique()
    
    # Get the timestamps and crossponding number of satellites in the navigation data.
    timeSVMap = {time : len(navigation_data.loc[time]) for time in navigation_data.index.get_level_values("time").unique()}
    
    # Match the timestamps based on the mode.
    matchesTimestamps = []
    for obsTime in obsTimestamps:
        # Get the timestamps within the threshold.
        navTimes = [time for time in timeSVMap.keys() if abs(time - obsTime) <= matching_threshold]
        
        # If no timestamps are found, skip.
        if len(navTimes) == 0:
            raise ValueError(f"No Navigation Data found for {obsTime} within +- 3 hours.")
        
        if mode.lower() == "maxsv":
            # Return the timestamp with the maximum number of satellites.
            matchesTimestamps.append((obsTime, max(navTimes, key=lambda time: timeSVMap[time])))         
        elif mode.lower() == "nearest":
            # Return the timestamp with the nearest timestamp.
            matchesTimestamps.append((obsTime, min(navTimes, key=lambda time: abs(time - obsTime))))
        else:
            raise ValueError(f"Mode must be in ['maxsv', 'nearest']. Got {mode}.")
        
    return matchesTimestamps
          
# TODO: Add the compatibility for other GNSS systems
def from_rinex_dataframes(
    observation_data: pd.DataFrame,
    observation_metadata: pd.Series,
    navigation_data: pd.DataFrame,
    navigation_metadata: pd.Series,
    station_name: tp.Optional[str] = None,
    matching_mode: str = 'maxsv',
    trim: bool = True,
    drop_na: bool = True,
    column_mapper: tp.Optional[tp.List[str]] = None,
    matching_threshold: pd.Timedelta = pd.Timedelta(hours=3),
    profile : dict[str, str| bool] = Epoch.INITIAL,
) -> tp.Iterator[Epoch]:
    """Loads the RINEX data to an epoch object.

    This methods assumes that the observation file is within one unique day.
    If the observation file contains data from multiple days, the split the data
    first and then load the data.

    Args:
        observation_data (pd.DataFrame): The observation data.
        observation_metadata (pd.Series): The observation metadata.
        navigation_data (pd.DataFrame): The navigation data.
        navigation_metadata (pd.Series): The navigation metadata.
        station_name (str, optional): The station name. Defaults to None.
        matching_mode (str, optional): The mode of matching the navigation data.[maxsv | nearest]. Defaults to 'maxsv'.
        trim (bool, optional): Intersect satellite vehicles in observation and navigation data. Defaults to True.
        drop_na (bool, optional): If True, the NaN values will be dropped from relevant columns. Defaults to True.
        column_mapper (tp.List[str], optional): The column mapper. Defaults to None.
        matching_threshold (pd.Timedelta, optional): The matching threshold to match the observation and navigation data. Defaults to pd.Timedelta(hours=3).
        profile (dict[str, str| bool], optional): The profile of the epoch. Defaults to Epoch.INITIAL.
        
    Returns:
        tp.List[Epoch]: A list of epoch objects.
    """
    # Check if the data is empty
    if observation_data.empty:
        raise ValueError(
            "The observation data is empty. Could not parse the observation file."
        )

    # Get the matches timestamps
    matchesTimestamps = get_fragments_idx_from_nav(
        observation_data=observation_data,
        navigation_data=navigation_data,
        mode=matching_mode,
        matching_threshold=matching_threshold,
    )
    
    for obsTime, navTime in matchesTimestamps:
        # Make the epoch object
        yield Epoch(
            timestamp=obsTime,
            obs_data=observation_data.loc[obsTime],
            obs_meta=observation_metadata,
            nav_data=navigation_data.loc[[navTime]], # Do not drop the time index here since it is TOC
            nav_meta=navigation_metadata,
            trim=trim,
            purify=drop_na,
            station=station_name,
            columns_mapping=column_mapper,
            profile=profile,
        )

# # TODO: Add the compatibility for other GNSS systems
def from_precise_ephemeris(
    observation_data: pd.DataFrame,
    observation_metadata: pd.Series,
    sp3_data: pd.DataFrame,
    station_name: tp.Optional[str] = None,
    trim: bool = True,
    drop_na: bool = True,
    column_mapper: tp.Optional[tp.List[str]] = None,
) -> tp.Iterator[Epoch]:
    """Loads the observation data and precise ephemeris data to an epoch object.

    Args:
        observation_data (pd.DataFrame): The observation data.
        observation_metadata (pd.Series): The observation metadata.
        sp3_data (pd.DataFrame): The SP3 data.
        station_name (str, optional): The station name. Defaults to None.
        trim (bool, optional): Intersect satellite vehicles in observation and navigation data. Defaults to True.
        drop_na (bool, optional): If True, the NaN values will be dropped from relevant columns. Defaults to True.
        column_mapper (tp.List[str], optional): The column mapper. Defaults to None.
        matching_threshold (pd.Timedelta, optional): The matching threshold to interpolate the sp3 data. Defaults to pd.Timedelta(hours=2).

    Returns:
        tp.List[Epoch]: A list of epoch objects.
    """
    # Check if the data is empty
    if observation_data.empty:
        raise ValueError(
            "The observation data is empty. Could not parse the observation file."
        )

    # Fragment the observation data
    obsTimes : list[pd.Timestamp] = observation_data.index.get_level_values("time").unique()
    
    for t in obsTimes:
        # Reference the sp3 data to the epoch
        nav_data = sp3_data

        yield Epoch(
            timestamp=t,
            obs_data=observation_data.loc[t],
            obs_meta=observation_metadata,
            nav_data=nav_data,
            station=station_name,
            trim=trim,
            purify=drop_na,
            profile=Epoch.SP3,
            columns_mapping=column_mapper,
        )

def from_rinex_files(
    observation_file: Path,
    navigation_file: Path,
    station_name: tp.Optional[str] = None,
    mode: str = 'maxsv',
    trim: bool = True,
    drop_na: bool = True,
    column_mapper: tp.Optional[tp.List[str]] = None,
    matching_threshold: pd.Timedelta = pd.Timedelta(hours=3),
    profile : dict[str, str| bool] = Epoch.INITIAL,
) -> tp.Iterator[Epoch]:
    """Loads the RINEX files to an epoch object.

    Args:
        observation_file (Path): The observation file.
        navigation_file (Path): The navigation file.
        station_name (str, optional): The station name. Defaults to None.
        mode (str, optional): The mode of matching the navigation data.[maxsv | nearest]. Defaults to 'maxsv'.
        trim (bool, optional): Intersect satellite vehicles in observation and navigation data. Defaults to True.
        drop_na (bool, optional): If True, the NaN values will be dropped from relevant columns. Defaults to True.
        column_mapper (tp.List[str], optional): The column mapper. Defaults to None.
        matching_threshold (pd.Timedelta, optional): The matching threshold to match the observation and navigation data. Defaults to pd.Timedelta(hours=3).
        profile (dict[str, str| bool], optional): The profile of the epoch. Defaults to Epoch.INITIAL.
        
    Returns:
        tp.List[Epoch]: A list of epoch objects.
    """
    # Load the data
    obsParser = IParseGPSObs()
    navParser = IParseGPSNav()

    # Load the observation data
    obsmeta, obsdata = obsParser.parse(observation_file)

    # Load the navigation data
    navmeta, navdata = navParser.parse(navigation_file)

    return from_rinex_dataframes(
        observation_data=obsdata,
        observation_metadata=obsmeta,
        navigation_data=navdata,
        navigation_metadata=navmeta,
        station_name=station_name,
        matching_mode=mode,
        trim=trim,
        drop_na=drop_na,
        column_mapper=column_mapper,
        matching_threshold=matching_threshold,
        profile=profile,
    )
