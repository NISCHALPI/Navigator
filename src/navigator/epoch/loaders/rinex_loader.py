"""Load the rinex data to an epoch object."""

import typing as tp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ...epoch.epoch import Epoch
from ...parse.iparse.nav.iparse_gps_nav import IParseGPSNav
from ...parse.iparse.obs.iparse_gps_obs import IParseGPSObs
from .fetchers import fetch_sp3
from .fragments.gps_framgents import FragNav, FragObs

__all__ = [
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
def from_rinex_dataframes(
    observation_data: pd.DataFrame,
    observation_metadata: pd.Series,
    navigation_data: pd.DataFrame,
    navigation_metadata: pd.Series,
    station_name: tp.Optional[str] = None,
    mode: str = 'maxsv',
    trim: bool = True,
    drop_na: bool = True,
    column_mapper: tp.Optional[tp.List[str]] = None,
    matching_threshold: pd.Timedelta = pd.Timedelta(hours=3),
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
        mode (str, optional): The mode of matching the navigation data.[maxsv | nearest]. Defaults to 'maxsv'.
        trim (bool, optional): Intersect satellite vehicles in observation and navigation data. Defaults to True.
        drop_na (bool, optional): If True, the NaN values will be dropped from relevant columns. Defaults to True.
        column_mapper (tp.List[str], optional): The column mapper. Defaults to None.
        matching_threshold (pd.Timedelta, optional): The matching threshold to match the observation and navigation data. Defaults to pd.Timedelta(hours=3).

    Returns:
        tp.List[Epoch]: A list of epoch objects.
    """
    # Check if the data is empty
    if observation_data.empty:
        raise ValueError(
            "The observation data is empty. Could not parse the observation file."
        )

    # Fragment the observation data
    observational_fragments: list[FragObs] = FragObs.fragmentify(
        obs_data=observation_data,
        station=station_name,
        obs_meta=observation_metadata,
    )
    navigation_data: list[FragNav] = FragNav.fragmentify(
        nav_data=navigation_data,
        station=station_name,
        nav_meta=navigation_metadata,
    )

    for obs_frags in observational_fragments:
        # Get the nearest navigation fragment
        optimal_nav_frag = obs_frags.nearest_nav_fragment(
            nav_fragments=navigation_data,
            mode=mode,
            matching_threshold=matching_threshold,
        )
        yield Epoch(
            timestamp=obs_frags.epoch_time,
            obs_data=obs_frags.obs_data,
            obs_meta=obs_frags.metadata,
            nav_data=optimal_nav_frag.nav_data,
            nav_meta=optimal_nav_frag.metadata,
            trim=trim,
            purify=drop_na,
            station=obs_frags.station_name if station_name != "CUSTOM" else None,
            columns_mapping=column_mapper,
            profile=Epoch.INITIAL,
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
    observational_fragments: list[FragObs] = FragObs.fragmentify(
        obs_data=observation_data,
        station=station_name,
        obs_meta=observation_metadata,
    )

    for obs_frags in observational_fragments:
        # Reference the sp3 data to the epoch
        nav_data = sp3_data

        yield Epoch(
            timestamp=obs_frags.epoch_time,
            obs_data=obs_frags.obs_data,
            obs_meta=obs_frags.metadata,
            nav_data=nav_data,
            nav_meta=pd.Series(),
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
        mode=mode,
        trim=trim,
        drop_na=drop_na,
        column_mapper=column_mapper,
        matching_threshold=matching_threshold,
    )
