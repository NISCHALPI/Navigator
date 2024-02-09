"""This module defines the Epoch class.

The Epoch class represents an Epoch instance with observational and navigation data fragments.
This class provides functionalities to handle and process data related to a single epoch,
including observational and navigation data, trimming, purifying, saving, loading, and more.

Classes:
    Epoch: Represents an Epoch instance with observational and navigation data fragments.

Attributes:
    __all__ (list[str]): List of module level attributes, classes and functions.

Author:
Name- Nischal Bhattari
Email- nischalbhattaraipi@gmail.com
"""

import pickle
from copy import deepcopy
from pathlib import Path  # type: ignore
from typing import Iterator

import pandas as pd  # type: ignore

from ..parse import Parser
from ..parse.iparse import IParseGPSNav, IParseGPSObs
from .epochfragment import FragNav, FragObs

__all__ = ["Epoch"]


class Epoch:
    """Represents an Epoch instance with observational and navigation data fragments.

    This class provides functionalities to handle and process data related to a single epoch,
    including observational and navigation data, trimming, purifying, saving, loading, and more.

    Attributes:
        _obs_frag (FragObs): Observational data fragment for a single epoch.
        _nav_frag (FragNav): Navigation data fragment for the corresponding epoch.

    Methods:
        __init__: Initialize an Epoch instance.
        timestamp: Get the timestamp of the epoch.
        obs_data: Get the observational data of the epoch.
        obs_data.setter: Set the observational data of the epoch.
        nav_data: Get the navigation data of the epoch.
        nav_data.setter: Set the navigation data of the epoch.
        trim: Intersect satellite vehicles in observation and navigation data.
        purify: Remove observations with missing data.
        common_sv: Get common satellite vehicles between observation and navigation data.
        __repr__: Return a string representation of the Epoch.
        __getitem__: Retrieve observables for a specific satellite vehicle by index.
        __len__: Return the number of satellite vehicles in the epoch.
        epochify: Generate Epoch instances from observation and navigation data files.
        save: Save the epoch to a file.
        load: Load an epoch from a file.
        load_from_fragment: Load an epoch from fragments.
        load_from_fragment_path: Load an epoch from fragment paths.
    """

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
        """
        # Store FragObs and FragNav
        self._obs_frag = deepcopy(
            obs_frag
        )  # Need to deepcopy to avoid modifying the original object
        self._nav_frag = deepcopy(
            nav_frag
        )  # Need to deepcopy to avoid modifying the original object

        # Flags
        self._is_smoothed = False

        # Purify the data
        if purify:
            self.obs_data = self.purify(self.obs_data)

        if trim:
            self.trim()

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
    def is_smoothed(self) -> bool:
        """Check if the epoch is smoothed.

        Returns:
            bool: True if the epoch is smoothed, False otherwise.
        """
        return self._is_smoothed

    @is_smoothed.setter
    def is_smoothed(self, value: bool) -> None:
        """Set the is_smoothed attribute.

        Args:
            value (bool): The value to set.

        """
        self._is_smoothed = value

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
    def epochify(obs: Path, nav: Path, mode: str = "maxsv") -> Iterator["Epoch"]:
        """Generate Epoch instances from observation and navigation data files.

        Args:
            obs (Path): Path to the observation data file.
            nav (Path): Path to the navigation data file.
            mode (str, optional): Ephemeris method. Either 'nearest' or 'maxsv'. Defaults to 'maxsv'.

        Yields:
            Iterator[Epoch]: Epoch instances generated from the provided data files.
        """
        # Parse the observation and navigation data
        parser = Parser(iparser=IParseGPSNav())

        # Parse the navigation data
        nav_meta, nav_data = parser(nav)

        if nav_data.empty:
            raise ValueError("No navigation data found.")

        # Parse the observation data
        parser.swap(iparser=IParseGPSObs())

        # Parse the observation data
        obs_meta, data = parser(obs)

        if data.empty:
            raise ValueError("No observation data found.")

        # Get the observational fragments
        obs_frags = FragObs.fragmentify(
            obs_data=data, parent=obs.name, obs_meta=obs_meta
        )

        # Get the navigation fragments
        nav_frags = FragNav.fragmentify(
            nav_data=nav_data, parent=nav.name, nav_meta=nav_meta
        )

        # Filter at least 4 satellites
        obs_frags = [frag for frag in obs_frags if len(frag) >= 4]
        nav_frags = [frag for frag in nav_frags if len(frag) >= 4]

        # Iterate over the fragments
        for observational_fragments in obs_frags:
            nearest_nav = observational_fragments.nearest_nav_fragment(
                nav_fragments=nav_frags, mode=mode
            )
            if nearest_nav is None:
                continue

            yield Epoch.load_from_fragment(
                obs_frag=observational_fragments,
                nav_frag=nearest_nav,
            )

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
