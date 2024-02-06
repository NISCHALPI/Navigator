"""Module to handle fragments of an epoch - observation and navigation data.

Useful for handling large files with many epochs, and for saving and loading space.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ..matcher.matcher import GpsNav3DailyMatcher, MixedObs3DailyMatcher

__all__ = ["Fragment", "FragObs", "FragNav"]


class Fragment(ABC):
    """Abstract base class for epoch fragments.

    Attributes:
        parent (str): Parent file name.
    """

    def __init__(self, parent: str) -> None:
        """Initialize a Fragment object.

        Args:
            parent (str): Parent file name.
        """
        self.parent = parent

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the fragment.

        Returns:
            str: Fragment name.
        """
        pass

    def save(self, save_path: Path) -> None:
        """Save the fragment to a file.

        Args:
            save_path (Path): Path to save the fragment.
        """
        with open(save_path / self.get_name(), "wb") as file:
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
        return f"{self.__class__.__name__}(parent='{self.parent}')"


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
        parent: str,
        metadata: pd.Series = None,
    ) -> None:
        """Initialize a FragObs object.

        Args:
            epoch_time (pd.Timestamp): Fragment epoch time.
            obs_data (pd.DataFrame): Fragment observation data.
            parent (str): Parent file name.
            metadata (pd.Series): Observation metadata.
        """
        super().__init__(parent)
        self.obs_data = obs_data
        self.epoch_time = epoch_time
        self.metadata = metadata

    def get_name(self) -> str:
        """Get the name of the fragment.

        Returns:
            str: Fragment name.
        """
        # Get the matcher
        matcher = MixedObs3DailyMatcher()

        if not matcher(self.parent):
            raise ValueError(
                "Parent file is not an observation file format defined by nasa cddis."
            )
        # Get the metadata from the parent file name.
        met = matcher.extract_metadata(self.parent)

        # Return the fragment name.
        return f"OBSFRAG_{met['station_name']}_{self.epoch_time.strftime('%Y%m%d_%H%M%S')}.pkl"

    def fragmentify(
        obs_data: pd.DataFrame, parent: str, obs_meta: pd.Series = None
    ) -> list["FragObs"]:
        """Fragmentify the observation data.

        Args:
            obs_data (pd.DataFrame): Observation data.
            parent (str): Parent file name.
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
            fragment = FragObs(timestamp, data, parent, obs_meta)

            # Append the fragment to the list.
            fragments.append(fragment)

        return fragments

    def nearest_nav_fragment(
        self, nav_fragments: list["FragNav"], mode: str = "maxsv"
    ) -> "FragNav":
        """Get the nearest navigation fragment to the observation.

        Args:
            nav_fragments (list[FragNav]): List of navigation fragments.
            mode (str, optional): Mode to get the nearest navigation fragment (maxsv | nearest). Defaults to "max_sv".

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
            if abs(fragment.timestamp - self.epoch_time) <= pd.Timedelta(hours=3)
        ]

        # If no fragments are found, return None.
        if len(nav_fragments) == 0:
            return None

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
    def station(self) -> str:
        """Get the station name.

        Returns:
            str: Station name.
        """
        return self.get_name().split("_")[1]

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
        parent (str): Parent file name.
    """

    def __init__(
        self,
        timestamp: pd.Timestamp,
        nav_data: pd.DataFrame,
        parent: str,
        metadata: pd.Series = None,
    ) -> None:
        """Initialize a FragNav object.

        Args:
            timestamp (pd.Timestamp): Fragment timestamp.
            nav_data (pd.DataFrame): Fragment navigation data.
            parent (str): Parent file name.
            metadata (pd.Series): Navigation metadata.
        """
        super().__init__(parent)
        self.timestamp = timestamp
        self.nav_data = nav_data
        self.metadata = metadata

    def get_name(self) -> str:
        """Get the name of the fragment.

        Returns:
            str: Fragment name.
        """
        # Get the matcher
        matcher = GpsNav3DailyMatcher()

        # Check if the parent file is a navigation file.
        if not matcher(self.parent):
            raise ValueError(
                "Parent file is not a navigation file format defined by nasa cddis."
            )

        # Get the metadata from the parent file name.
        met = matcher.extract_metadata(self.parent)

        # Return the fragment name.
        return f"NAVFRAG_{met['station_name']}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.pkl"

    @staticmethod
    def fragmentify(
        nav_data: pd.DataFrame, parent: str, nav_meta: pd.Series
    ) -> list["FragNav"]:
        """Fragmentify the navigation data.

        Args:
            nav_data (pd.DataFrame): Navigation data.
            parent (str): Parent file name.
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
            fragment = FragNav(timestamp, data, parent, nav_meta)

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
