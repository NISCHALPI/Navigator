"""Module to handle fragments of dataframe parsed by georinex backend library.

Useful for handling large files with many epochs, and for saving and loading space.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

__all__ = ["Fragment", "FragObs", "FragNav"]


class Fragment(ABC):
    """Abstract base class for epoch fragments.

    Attributes:
        parent (str): Parent file name.
    """

    def __init__(self, station_name: str = "CUSTOM") -> None:
        """Initialize a Fragment object.

        Args:
            station_name (str): IGS station name.
        """
        self.station_name = station_name

    @abstractmethod
    def fragmentify(data: pd.DataFrame, station: str, meta: pd.Series = None) -> list:
        """Fragmentify the data.

        Args:
            data (pd.DataFrame): Data to fragmentify.
            station (str): Parent file name.
            meta (pd.Series): Metadata.

        Returns:
            list: List of fragments.
        """
        pass

    def save(self, save_path: Path) -> None:
        """Save the fragment to a file.

        Args:
            save_path (Path): Path to save the fragment.
        """
        with open(save_path / self.station_name, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(load_path: Path) -> "Fragment":
        """Load the fragment from a file.

        Args:
            load_path (Path): Path to load the fragment.

        Returns:
            Fragment: Loaded fragment object.
        """
        with open(load_path, "rb") as file:
            fragment = pickle.load(file)

        if not isinstance(fragment, Fragment):
            raise TypeError("Loaded object is not a Fragment.")

        return fragment

    def __repr__(self) -> str:
        """Get the representation of the Fragment object.

        Returns:
            str: Representation of the Fragment object.
        """
        return f"{self.__class__.__name__}(parent='{self.station_name}')"


class FragObs(Fragment):
    """Represents a fragment of an epoch containing observation data.

    Attributes:
        epoch_time (pd.Timestamp): Fragment epoch time.
        obs_data (pd.DataFrame): Fragment observation data.
        parent (str): Parent file name.
    """

    def __init__(
        self,
        epoch_time: pd.Timestamp,
        obs_data: pd.DataFrame,
        station_name: str,
        metadata: pd.Series = None,
    ) -> None:
        """Initialize a FragObs object.

        Args:
            epoch_time (pd.Timestamp): Fragment epoch time.
            obs_data (pd.DataFrame): Fragment observation data.
            station_name (str): IGS station name.
            metadata (pd.Series): Observation metadata.
        """
        super().__init__(
            station_name=station_name,
        )
        self.obs_data = obs_data
        self.epoch_time = epoch_time
        self.metadata = metadata

    def fragmentify(
        obs_data: pd.DataFrame, station: str, obs_meta: pd.Series = None
    ) -> list["FragObs"]:
        """Fragmentify the observation data.

        Args:
            obs_data (pd.DataFrame): Observation data.
            station (str): Station name.
            obs_meta (pd.Series): Observation metadata.


        Returns:
            list[FragObs]: List of fragments.
        """
        # Get the timestamps of the observations.
        timestamps = obs_data.index.get_level_values("time").unique()

        # For each timestamp, create a fragment.
        fragments = []

        for timestamp in timestamps:
            # Get the data for the current timestamp
            data = obs_data.xs(key=timestamp, level="time", drop_level=True)

            # Create the fragment.
            fragment = FragObs(timestamp, data, station, obs_meta)

            # Append the fragment to the list.
            fragments.append(fragment)

        return fragments

    def nearest_nav_fragment(
        self,
        nav_fragments: list["FragNav"],
        mode: str = "maxsv",
        matching_threshold: pd.Timedelta = pd.Timedelta(hours=3),
    ) -> "FragNav":
        """Get the nearest navigation fragment to the observation.

        Args:
            nav_fragments (list[FragNav]): List of navigation fragments.
            mode (str, optional): Mode to get the nearest navigation fragment (maxsv | nearest). Defaults to "max_sv".
            matching_threshold (pd.Timedelta, optional): Matching threshold for the observation time. Defaults to pd.Timedelta(hours=3).

        Returns:
            FragNav: Nearest navigation fragment.

        Raises:
            ValueError: Mode must be either maxsv or nearest.
        """
        # Mode must be either max_sv or nearest.
        if mode not in ["maxsv", "nearest"]:
            raise ValueError("Mode must be either maxsv or nearest.")

        # Filter to +- 3 hours of the observation time.
        nav_fragments = [
            fragment
            for fragment in nav_fragments
            if abs(fragment.timestamp - self.epoch_time) <= matching_threshold
        ]

        # If no fragments are found, return None.
        if len(nav_fragments) == 0:
            raise ValueError(
                f"No Navigation Data found for {self.epoch_time} within +- 3 hours."
            )

        if mode == "maxsv":
            # Return the fragment with the maximum number of satellites in common.
            return max(
                nav_fragments,
                key=lambda fragment: len(fragment.intersect_sv(self)),
            )
        if mode == "nearest":
            # Return the fragment with the nearest timestamp.
            return min(
                nav_fragments,
                key=lambda fragment: abs(fragment.timestamp - self.epoch_time),
            )
        return None

    @property
    def svs(self) -> pd.Index:
        """Get the satellite vehicles in the fragment.

        Returns:
            pd.Index: Satellite vehicles.
        """
        return self.obs_data.index.get_level_values("sv").unique()

    def __len__(self) -> int:
        """Get the length of the fragment.

        Returns:
            int: Length of the fragment.
        """
        return len(self.obs_data.index.get_level_values("sv").unique())


class FragNav(Fragment):
    """Represents a fragment of an epoch containing navigation data.

    Attributes:
        timestamp (pd.Timestamp): Fragment timestamp.
        nav_data (pd.DataFrame): Fragment navigation data.
        station_name (str): IGS station name.
    """

    def __init__(
        self,
        timestamp: pd.Timestamp,
        nav_data: pd.DataFrame,
        station_name: str,
        metadata: pd.Series = None,
    ) -> None:
        """Initialize a FragNav object.

        Args:
            timestamp (pd.Timestamp): Fragment timestamp.
            nav_data (pd.DataFrame): Fragment navigation data.
            station_name (str): IGS station name.
            metadata (pd.Series): Navigation metadata.
        """
        super().__init__(station_name=station_name)
        self.timestamp = timestamp
        self.nav_data = nav_data
        self.metadata = metadata

    @staticmethod
    def fragmentify(
        nav_data: pd.DataFrame, station: str, nav_meta: pd.Series
    ) -> list["FragNav"]:
        """Fragmentify the navigation data.

        Args:
            nav_data (pd.DataFrame): Navigation data.
            station (str): Parent file name.
            nav_meta (pd.Series): Navigation metadata.

        Returns:
            list[FragNav]: List of fragments.
        """
        # Get the timestamps of the observations.
        timestamps = nav_data.index.get_level_values("time").unique()

        # For each timestamp, create a fragment.
        fragments = []

        for timestamp in timestamps:
            # Get the data for the current timestamp
            data = nav_data.xs(key=timestamp, level="time", drop_level=False)

            # Create the fragment.
            fragment = FragNav(timestamp, data, station, nav_meta)

            # Append the fragment to the list.
            fragments.append(fragment)

        return fragments

    def __len__(self) -> int:
        """Get the length of the fragment.

        Returns:
            int: Length of the fragment.
        """
        return len(self.nav_data.index.get_level_values("sv").unique())

    def intersect_sv(self, obs: "FragObs") -> pd.Index:
        """Get the intersection of satellites between the navigation and observation fragments.

        Args:
            obs (FragObs): Observation fragment.

        Returns:
            pd.Index: Intersection of satellites.
        """
        return self.nav_data.index.get_level_values("sv").intersection(
            obs.obs_data.index.get_level_values("sv")
        )
