"""Epoch Class.

This module defines the Epoch class, representing an Epoch instance with observational and navigation data fragments. The class provides functionalities to handle and process data related to a single epoch, including observational and navigation data manipulation, trimming, purifying, saving, loading, and more.

Attributes:
    SINGLE (dict): Single Frequency Profile, specifying settings for single frequency mode.
    DUAL (dict): Dual Frequency Profile, specifying settings for dual frequency mode.
    INIT (dict): Initial Profile, specifying settings without applying any corrections.

Classes:
    Epoch: Represents an Epoch instance with observational and navigation data fragments.

Methods:
    - __init__: Initializes an Epoch instance with observational and navigation data fragments.
    - trim: Intersects satellite vehicles in observation and navigation data.
    - purify: Removes observations with missing data.
    - save: Saves the epoch to a file.
    - load: Loads an epoch from a file.
    - load_from_fragment: Loads an epoch from observational and navigation fragments.
    - load_from_fragment_path: Loads an epoch from paths to observational and navigation fragments.

Example:
    ```python
    from gnss_module import Epoch

    # Create an Epoch instance
    epoch = Epoch(obs_frag=obs_fragment, nav_frag=nav_fragment, trim=True, purify=True)

    # Save the Epoch
    epoch.save("path/to/save/epoch.pkl")

    # Load the Epoch
    loaded_epoch = Epoch.load("path/to/save/epoch.pkl")
    ```

Attributes:
    - timestamp: Get the timestamp of the epoch.
    - obs_data: Get or set the observational data of the epoch.
    - nav_data: Get or set the navigation data of the epoch.
    - station: Get the station name.
    - nav_meta: Get the navigation metadata.
    - obs_meta: Get the observation metadata.
    - profile: Get or set the profile of the epoch.
    - is_smoothed: Get or set the smoothed attribute.

Methods:
    - __getitem__: Retrieve observables for a specific satellite vehicle (SV) by index.
    - __len__: Return the number of satellite vehicles (SVs) in the epoch.
    - __repr__: Return a string representation of the Epoch.

Author:
    Nischal Bhattarai
"""


import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path  # type: ignore
from tempfile import TemporaryDirectory
from typing import Iterator

import pandas as pd  # type: ignore

from ..download.idownload.rinex.nasa_cddis import NasaCddisV3
from ..parse import Parser
from ..parse.iparse import IParseGPSNav, IParseGPSObs
from .epochfragment import FragNav, FragObs

__all__ = ["Epoch"]


class Epoch:
    """Represents an Epoch instance with observational and navigation data fragments.

    This class provides functionalities to handle and process data related to a single epoch,
    including observational and navigation data, trimming, purifying, saving, loading, and more.


    """

    # Single Frequency Profile
    SINGLE = {
        "apply_tropo": True,
        "apply_iono": True,
        "mode": "single",
    }
    # Dual Frequency Profile
    DUAL = {
        "apply_tropo": True,
        "apply_iono": False,  # Iono free combination is used
        "mode": "dual",
    }
    # Initial Profile [Doesn't apply any corrections]
    INITIAL = {
        "apply_tropo": False,
        "apply_iono": False,
        "mode": "single",
    }

    def __init__(
        self,
        obs_frag: FragObs,
        nav_frag: FragNav,
        trim: bool = False,
        purify: bool = False,
    ) -> None:
        """Initialize an Epoch instance with a timestamp and observational data and navigation data.

        Args:
            timestamp (pd.Timestamp): The timestamp associated with the epoch.
            obs_frag (FragObs): A Observational data fragment i.e observational data for a single epoch.
            nav_frag (FragNav): A Navigation data fragment i.e navigation data for crossponding epoch.
            trim (bool, optional): Whether to trim the data. Defaults to False.
            purify (bool, optional): Whether to purify the data. Defaults to False.

        Returns:
            None
        """
        # Store FragObs and FragNav
        self._obs_frag = deepcopy(
            obs_frag
        )  # Need to deepcopy to avoid modifying the original object
        self._nav_frag = deepcopy(
            nav_frag
        )  # Need to deepcopy to avoid modifying the original object

        # Purify the data
        if purify:
            self.obs_data = self.purify(self.obs_data)

        # Trim the data if required
        if trim:
            self.trim()

        # Set the is_smoothed attribute
        self._is_smoothed = False

        # Define a profile for the epoch can be [dual , single]
        self._profile = self.DUAL

    @property
    def timestamp(self) -> pd.Timestamp:
        """Get the timestamp of the epoch.

        Returns:
            pd.Timestamp: The timestamp associated with the epoch.

        """
        return self._obs_frag.epoch_time

    @property
    def obs_data(self) -> pd.DataFrame:
        """Get the observational data of the epoch.

        Returns:
            pd.DataFrame: A DataFrame containing observational data.

        """
        return self._obs_frag.obs_data

    @obs_data.setter
    def obs_data(self, data: pd.DataFrame) -> None:  # noqa: ARG002
        """Set the observational data of the epoch.

        Args:
            data (pd.DataFrame): The data to set.

        """
        self._obs_frag.obs_data = data

    @property
    def nav_data(self) -> pd.DataFrame:
        """Get the navigation data of the epoch.

        Returns:
            pd.DataFrame: A DataFrame containing navigation data.

        """
        return self._nav_frag.nav_data

    @nav_data.setter
    def nav_data(self, nav_data: pd.DataFrame) -> None:  # noqa: ARG002
        """Set the navigation data of the epoch.

        Args:
            nav_data (pd.DataFrame): The nav_data to set.

        """
        self._nav_frag.nav_data = nav_data

    @property
    def station(self) -> str:
        """Get the station name.

        Returns:
            str: The station name.

        """
        return self._obs_frag.station

    @property
    def nav_meta(self) -> pd.Series:
        """Get the navigation metadata.

        Returns:
            pd.Series: The navigation metadata.

        """
        return self._nav_frag.metadata

    @property
    def obs_meta(self) -> pd.Series:
        """Get the observation metadata.

        Returns:
            pd.Series: The observation metadata.

        """
        return self._obs_frag.metadata

    @property
    def profile(self) -> dict:
        """Get the profile of the epoch.

        Returns:
            dict: The profile of the epoch.

        """
        return self._profile

    @profile.setter
    def profile(self, value: dict) -> None:
        """Set the profile of the epoch.

        Args:
            value (dict): The value to set.

        """
        # Necessary keys
        necessary_keys = ["apply_tropo", "apply_iono", "mode"]

        # Check if the value contains the necessary keys
        if not all(key in value for key in necessary_keys):
            raise ValueError(
                f"Profile must contain the following keys: {necessary_keys}. Got {value.keys()} instead."
            )
        self._profile = value

    @property
    def is_smoothed(self) -> bool:
        """Get the smoothed attribute.

        Returns:
            bool: True if the epoch has been smoothed, False otherwise.

        """
        return self.profile.get("smoothed", False)

    @is_smoothed.setter
    def is_smoothed(self, value: bool) -> None:
        """Set the smoothed attribute.

        Args:
            value (bool): The value to set.

        """
        self._profile["smoothed"] = value

    def trim(self) -> None:
        """Intersect the satellite vehicles in the observation data and navigation data.

        Trims the data to contain only satellite vehicles present in both observations and navigation.
        """
        # Drop the satellite vehicles not present in both observation and navigation data
        common_sv = self.common_sv.copy()
        self.obs_data = self.obs_data.loc[common_sv]
        self.nav_data = self.nav_data.loc[
            (slice(None), common_sv), :
        ]  # Nav data is multi-indexed
        return

    def purify(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove observations with missing data.

        Args:
            data (pd.DataFrame): DataFrame containing observational or navigation data.

        Returns:
            pd.DataFrame: DataFrame with missing data observations removed.
        """
        # Drop NA rows values for observations ["C1C", "C2C", "C2W" , "C1W"] if present
        if "C1C" in data.columns and "C2C" in data.columns:
            data = data.dropna(subset=["C1C", "C2C"], axis=0)
        elif "C1W" in data.columns and "C2W" in data.columns:
            data = data.dropna(subset=["C1W", "C2W"], axis=0)

        # Drop Duplicates columns
        return data[~data.index.duplicated(keep="first")]

    @property
    def common_sv(self) -> pd.Index:
        """Get the common satellite vehicles between the observation data and navigation data.

        Returns:
            pd.Index: Common satellite vehicles present in both observational and navigation data.
        """
        return self.obs_data.index.get_level_values("sv").intersection(
            self.nav_data.index.get_level_values("sv")
        )

    def __repr__(self) -> str:
        """Return a string representation of the Epoch.

        Returns:
            str: A string representation of the Epoch.

        """
        return f"Epoch(timestamp={self.timestamp}, sv={self.obs_data.shape[0]})"

    def __getitem__(self, sv: int) -> pd.Series:
        """Retrieve observables for a specific satellite vehicle (SV) by index.

        Args:
            sv (int): The index of the satellite vehicle (SV).

        Returns:
            pd.Series: A pandas Series containing observables for the specified SV.

        """
        return self.obs_data.loc[sv]

    def __len__(self) -> int:
        """Return the number of satellite vehicles (SVs) in the epoch.

        Returns:
            int: The number of satellite vehicles (SVs) in the epoch.

        """
        return len(self.obs_data)

    @staticmethod
    def _cddis_fetch(date: datetime) -> tuple[pd.Series, pd.DataFrame]:
        """Fetch the data from CDDIS for the given date.

        Args:
            date (datetime): The date to fetch the data for.

        Returns:
            tuple[pd.Series, pd.DataFrame]: A tuple containing the metadata as a Pandas Series and the parsed data as a Pandas DataFrame.
        """
        # Fetch the data from CDDIS
        downloader = NasaCddisV3(logging=True)
        parser = Parser(iparser=IParseGPSNav())

        # Set up an temporary directory

        with TemporaryDirectory() as temp_dir:
            # Download the navigation data
            downloader.download(
                year=date.year,
                day=date.timetuple().tm_yday,
                save_path=Path(temp_dir),
                num_files=1,
                match_string="JPL",  # Download from JPL Stations
            )

            # Get the navigation data file
            nav_file = list(Path(temp_dir).glob("*GN*"))[0]

            # Parse the navigation data
            nav_meta, nav_data = parser(nav_file)

        return nav_meta, nav_data

    @staticmethod
    def epochify(
        obs: Path, nav: Path | None = None, mode: str = "maxsv"
    ) -> Iterator["Epoch"]:
        """Generate Epoch instances from observation and navigation data files.

        Args:
            obs (Path): Path to the observation data file.
            nav (Path): Path to the navigation data file. Defaults to None.
            mode (str, optional): Ephemeris method. Either 'nearest' or 'maxsv'. Defaults to 'maxsv'.

        Yields:
            Iterator[Epoch]: Epoch instances generated from the provided data files.
        """
        # Parse the observation and navigation data
        parser = Parser(iparser=IParseGPSObs())

        # Parse the observation data
        obs_meta, data = parser(obs)

        if data.empty:
            raise ValueError("No observation data found.")

        if nav is not None:
            # Parse the observation data
            parser.swap(iparser=IParseGPSObs())
            # Parse the navigation data
            nav_meta, nav_data = parser(nav)
        else:
            # Fetch the navigation data from CDDIS
            try:
                nav_meta, nav_data = Epoch._cddis_fetch(
                    date=data.index[0][0]
                )  # Data is multi-indexed (time, sv)
            except Exception as e:
                raise ValueError(
                    f"Failed to fetch navigation data from CDDIS. Got the following error: {e}"
                )
        if nav_data.empty:
            raise ValueError("No navigation data found.")

        # Get the observational fragments
        obs_frags = FragObs.fragmentify(
            obs_data=data, parent=obs.name, obs_meta=obs_meta
        )

        # Get the navigation fragments
        nav_frags = FragNav.fragmentify(
            nav_data=nav_data,
            parent=nav.name if nav is not None else None,  # Guard against None
            nav_meta=nav_meta,
        )

        # Filter at least 4 satellites
        obs_frags = [frag for frag in obs_frags if len(frag) >= 4]
        nav_frags = [frag for frag in nav_frags if len(frag) >= 4]

        # Iterate over the fragments
        yield from [
            Epoch.load_from_fragment(
                obs_frag=observational_fragments, nav_frag=nearest_nav
            )
            for observational_fragments in obs_frags
            if (
                nearest_nav := observational_fragments.nearest_nav_fragment(
                    nav_fragments=nav_frags, mode=mode
                )
            )
            is not None
        ]

    def save(self, path: str | Path) -> None:
        """Save the epoch to a file.

        Args:
            path (str): The path to save the epoch to.

        Returns:
            None

        """
        # Pickle the epoch object
        with open(path, "wb") as file:
            pickle.dump(self, file)

        return

    @staticmethod
    def load(path: str | Path) -> "Epoch":
        """Load an epoch from a file.

        Args:
            path (str): The path to load the epoch from.

        Returns:
            Epoch: The epoch loaded from the file.

        """
        # Unpickle the epoch object
        with open(path, "rb") as file:
            epoch = pickle.load(file)

        # Check if the loaded object is an Epoch
        if not isinstance(epoch, Epoch):
            raise TypeError(
                f"Loaded object is not an Epoch. Got {type(epoch)} instead."
            )

        return epoch

    @staticmethod
    def load_from_fragment(obs_frag: FragObs, nav_frag: FragNav) -> "Epoch":
        """Load an epoch from fragments.

        Args:
            obs_frag (list[FragObs]): Observation data fragments.
            nav_frag (pd.DataFrame): Navigation data fragments.

        Returns:
            None
        """
        return Epoch(obs_frag=obs_frag, nav_frag=nav_frag, trim=True, purify=True)

    @staticmethod
    def load_from_fragment_path(obs_frag_path: Path, nav_frag_path: Path) -> "Epoch":
        """Load an epoch from fragments.

        Args:
            obs_frag_path (Path): Path to the observation fragment.
            nav_frag_path (Path): Path to the navigation fragment.

        Returns:
            Epoch: The epoch loaded from the fragments.
        """
        return Epoch.load_from_fragment(
            obs_frag=FragObs.load(obs_frag_path), nav_frag=FragNav.load(nav_frag_path)
        )

    def __gt__(self, other: "Epoch") -> bool:
        """Check if the epoch is greater than another epoch.

        Args:
            other (Epoch): The other epoch to compare to.

        Returns:
            bool: True if the epoch is greater than the other epoch, False otherwise.
        """
        # If not same station, raise error
        if self.station != other.station:
            raise ValueError(
                f"Cannot compare epochs with different stations. Got {self.station} and {other.station}."
            )
        return self.timestamp > other.timestamp

    def __lt__(self, other: "Epoch") -> bool:
        """Check if the epoch is less than another epoch.

        Args:
            other (Epoch): The other epoch to compare to.

        Returns:
            bool: True if the epoch is less than the other epoch, False otherwise.
        """
        # If not same station, raise error
        if self.station != other.station:
            raise ValueError(
                f"Cannot compare epochs with different stations. Got {self.station} and {other.station}."
            )
        return self.timestamp < other.timestamp

    def __eq__(self, other: "Epoch") -> bool:
        """Check if the epoch is equal to another epoch (same timestamp and station).

        Args:
            other (Epoch): The other epoch to compare to.

        Returns:
            bool: True if the epoch is equal to the other epoch, False otherwise.
        """
        # If not same station, raise error
        if self.station != other.station:
            raise ValueError(
                f"Cannot compare epochs with different stations. Got {self.station} and {other.station}."
            )
        return self.timestamp == other.timestamp and self.station == other.station
