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
from pathlib import Path  # type: ignore

import pandas as pd  # type: ignore

from ..utils.igs_network import IGSNetwork

__all__ = ["Epoch"]


class Epoch:
    """Represents an Epoch instance with observational and navigation data fragments.

    This class provides functionalities to handle and process data related to a single epoch,
    including observational and navigation data, trimming, purifying, saving, loading, and more.


    """

    # RINEX Observables
    L1_CODE_ON = "C1C"
    L2_CODE_ON = "C2W"
    L1_PHASE_ON = "L1C"
    L2_PHASE_ON = "L2W"

    OBSERVABLES = [L1_CODE_ON, L2_CODE_ON, L1_PHASE_ON, L2_PHASE_ON]

    # Relvant columns
    MINUMUM_REQUIRED_COLUMNS_MAP = {
        "single": [L1_CODE_ON, L1_PHASE_ON],
        "dual": [L1_CODE_ON, L2_CODE_ON, L1_PHASE_ON, L2_PHASE_ON],
        "phase": [L1_CODE_ON, L2_CODE_ON, L1_PHASE_ON, L2_PHASE_ON],
    }

    # Single Frequency Profile
    SINGLE = {
        "apply_tropo": True,
        "apply_iono": True,
        "mode": "single",
        "smoothed": False,
        "navigation_format": "ephemeris",
        "tropospheric_model": "unb3m",
    }
    # Dual Frequency Profile
    DUAL = {
        "apply_tropo": True,
        "apply_iono": True,
        "mode": "dual",
        "smoothed": False,
        "navigation_format": "ephemeris",
        "tropospheric_model": "unb3m",
    }
    # Initial Profile [Doesn't apply any corrections]
    INITIAL = {
        "apply_tropo": False,
        "apply_iono": False,
        "mode": "single",
        "smoothed": False,
        "navigation_format": "ephemeris",
        "tropospheric_model": "saastamoinen",
    }
    # SP3 Profile
    SP3 = {
        "apply_tropo": False,
        "apply_iono": False,
        "mode": "single",
        "smoothed": True,
        "navigation_format": "sp3",
        "tropospheric_model": "saastamoinen",
    }
    # Phase Profile
    PHASE = {
        "apply_tropo": False,
        "apply_iono": False,
        "mode": "phase",
        "smoothed": False,
        "navigation_format": "ephemeris",
        "tropospheric_model": "saastamoinen",
    }

    # DUMMY
    DUMMY = {
        "apply_tropo": False,
        "apply_iono": False,
        "mode": "dummy",
        "smoothed": False,
        "navigation_format": "ephemeris",
        "tropospheric_model": "saastamoinen",
    }

    ALLOWED_PROFILE_KEYS = [
        "apply_tropo",
        "apply_iono",
        "mode",
        "smoothed",
        "navigation_format",
        "tropospheric_model",
    ]
    MANDATORY_PROFILE_KEYS = [
        "apply_tropo",
        "apply_iono",
        "mode",
        "navigation_format",
        "tropospheric_model",
    ]

    # IGS network
    IGS_NETWORK = IGSNetwork()

    # Mandatory keys for real coordinates
    MANDATORY_COORDS_KEYS = ["x", "y", "z"]

    # Ionospheric Correction Keys
    IONOSPHERIC_KEY = "IONOSPHERIC CORR"

    def __init__(
        self,
        timestamp: pd.Timestamp,
        obs_data: pd.DataFrame,
        nav_data: pd.DataFrame,
        obs_meta: pd.Series,
        nav_meta: pd.Series,
        trim: bool = False,
        purify: bool = False,
        real_coord: pd.Series | dict | None = None,
        approximate_coords: pd.Series | dict | None = None,
        station: str | None = None,
        columns_mapping: dict | None = None,
        profile: dict[str, bool | str] = INITIAL,
    ) -> None:
        """Initialize an Epoch instance with a timestamp and observational data and navigation data.

        Args:
            timestamp (pd.Timestamp): The timestamp of the epoch i.e reciever timestamp.
            obs_data (pd.DataFrame): The observational data of the epoch.
            obs_meta (pd.Series): The observation metadata.
            nav_data (pd.DataFrame): The navigation data of the epoch.
            nav_meta (pd.Series): The navigation metadata.
            trim (bool, optional): Intersect satellite vehicles in observation and navigation data. Defaults to False.
            purify (bool, optional): Remove observations with missing data. Defaults to False.
            real_coord (pd.Series | dict | None, optional): The real coordinates of the station. Defaults to None.
            approximate_coords (pd.Series | dict | None, optional): The approximate coordinates of the station. Defaults to None.
            station (str | None, optional): The station name. Defaults to None.
            columns_mapping (dict | None, optional): The columns mapping to map names of columns to appropriate definitions. Defaults to None.
            profile (dict[str, bool | str], optional): The profile of the epoch. Defaults to INITIAL.

        Returns:
            None

        Note:
            - The code and phase observables are expected to be in the following format:
                - L1 Code: 'C1C'
                - L2 Code: 'C2W'
                - L1 Phase: 'L1C'
                - L2 Phase: 'L2W'
            - If the name of the columns are different, a columns_mapping dictionary can be provided to map the names to the appropriate definitions.
                for example:
                columns_mapping = {
                    'C1C': 'C1C',
                    'C2W': 'C2X',
                    'L1C': 'L1C',
                    'L2W': 'L2X'
                }
        """
        # Define a profile for the epoch can be [dual , single]
        self.profile = profile

        # Rename the colums to default names
        if columns_mapping is not None:
            # Reverse the dictionary
            columns_mapping = {v: k for k, v in columns_mapping.items()}
            obs_data = obs_data.rename(columns=columns_mapping)

        # Purify the data if required
        obs_data = (
            self.purify(
                obs_data,
                relevant_columns=Epoch.MINUMUM_REQUIRED_COLUMNS_MAP.get(
                    self.profile["mode"]
                ),
            )
            if purify
            else obs_data
        )

        # Store the timestamp
        self.timestamp = timestamp

        # Store FragObs and FragNav
        self.obs_data = obs_data
        self.nav_data = nav_data

        # Store the metadata
        self.obs_meta = obs_meta
        self.nav_meta = nav_meta

        # Trim the data if required
        if trim:
            self.trim()

        # Set the is_smoothed attribute
        self.is_smoothed = False

        # Set the real coordinates of the station
        self.real_coord = real_coord

        # Set the approximate coordinates of the station
        self.approximate_coords = approximate_coords

        # Set the station name
        self.station = station

        # Populate the IGS network if the station is part of the IGS network
        if self.station in self.IGS_NETWORK:
            self.real_coord = pd.Series(
                self.IGS_NETWORK.get_xyz(self._station), index=["x", "y", "z"]
            )

    @property
    def real_coord(self) -> pd.Series:
        """Get the real coordinates of the station.

        Returns:
            pd.Series: The real coordinates of the station.

        """
        return self._real_coord

    @real_coord.setter
    def real_coord(self, value: pd.Series | dict | None) -> None:
        """Set the real coordinates of the station.

        Args:
            value (pd.Series): The value to set.

        """
        if value is not None and not all(
            keys in value for keys in Epoch.MANDATORY_COORDS_KEYS
        ):
            raise ValueError(
                f"Real coordinates must contain the following keys: ['x', 'y', 'z']. Got {value.keys()} instead."
            )
        self._real_coord = pd.Series(value) if value is not None else pd.Series()

    @property
    def approximate_coords(self) -> pd.Series:
        """Get the approximate coordinates of the station.

        Returns:
            pd.Series: The approximate coordinates of the station.

        """
        return self._approximate_coords

    @approximate_coords.setter
    def approximate_coords(self, value: pd.Series | dict | None) -> None:
        """Set the approximate coordinates of the station.

        Args:
            value (pd.Series): The value to set.

        """
        if value is not None and not all(
            keys in value for keys in Epoch.MANDATORY_COORDS_KEYS
        ):
            raise ValueError(
                f"Approximate coordinates must contain the following keys: ['x', 'y', 'z']. Got {value.keys()} instead."
            )
        self._approximate_coords = (
            pd.Series(value) if value is not None else pd.Series()
        )

    @property
    def timestamp(self) -> pd.Timestamp:
        """Get the timestamp of the epoch.

        Returns:
            pd.Timestamp: The timestamp associated with the epoch.

        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: pd.Timestamp) -> None:
        """Set the timestamp of the epoch.

        Args:
            value (pd.Timestamp): The value to set.

        """
        if not isinstance(value, pd.Timestamp):
            raise ValueError(
                f"Timestamp must be a pd.Timestamp. Got {type(value)} instead."
            )

        self._timestamp = value

    @property
    def obs_data(self) -> pd.DataFrame:
        """Get the observational data of the epoch.

        Returns:
            pd.DataFrame: A DataFrame containing observational data.

        """
        return self._obs_data

    @obs_data.setter
    def obs_data(self, data: pd.DataFrame) -> None:  # noqa: ARG002
        """Set the observational data of the epoch.

        Args:
            data (pd.DataFrame): The data to set.

        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f"Observational data must be a pd.DataFrame. Got {type(data)} instead."
            )
        self._obs_data = data

    @property
    def nav_data(self) -> pd.DataFrame:
        """Get the navigation data of the epoch.

        Returns:
            pd.DataFrame: A DataFrame containing navigation data.

        """
        return self._nav_data

    @nav_data.setter
    def nav_data(self, nav_data: pd.DataFrame) -> None:  # noqa: ARG002
        """Set the navigation data of the epoch.

        Args:
            nav_data (pd.DataFrame): The nav_data to set.

        """
        if not isinstance(nav_data, pd.DataFrame):
            raise ValueError(
                f"Navigation data must be a pd.DataFrame. Got {type(nav_data)} instead."
            )
        self._nav_data = nav_data

    @property
    def station(self) -> str:
        """Get the station name.

        Returns:
            str: The station name.

        """
        return self._station

    @station.setter
    def station(self, value: str) -> None:
        """Set the station name.

        Args:
            value (str): The value to set.

        """
        self._station = value

    @property
    def nav_meta(self) -> pd.Series:
        """Get the navigation metadata.

        Returns:
            pd.Series: The navigation metadata.

        """
        return self._nav_meta

    @nav_meta.setter
    def nav_meta(self, value: pd.Series) -> None:
        """Set the navigation metadata.

        Args:
            value (pd.Series): The value to set.

        """
        if not isinstance(value, pd.Series):
            raise ValueError(
                f"Navigation metadata must be a pd.Series. Got {type(value)} instead."
            )
        self._nav_meta = value

    @property
    def obs_meta(self) -> pd.Series:
        """Get the observation metadata.

        Returns:
            pd.Series: The observation metadata.

        """
        return self._obs_meta

    @obs_meta.setter
    def obs_meta(self, value: pd.Series) -> None:
        """Set the observation metadata.

        Args:
            value (pd.Series): The value to set.

        """
        if not isinstance(value, pd.Series):
            raise ValueError(
                f"Observation metadata must be a pd.Series. Got {type(value)} instead."
            )
        self._obs_meta = value

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
        # Check if the value contains the necessary keys
        if not all(key in value for key in Epoch.MANDATORY_PROFILE_KEYS):
            raise ValueError(
                f"Profile must contain the following keys: {Epoch.MANDATORY_PROFILE_KEYS}. Got {value.keys()} instead."
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

        if self.profile["navigation_format"] == "ephemeris":
            # Ephemeris profile is multi-indexed
            self.nav_data = self.nav_data.loc[(slice(None), common_sv), :]
        elif self.profile["navigation_format"] == "sp3":
            # Do nothing
            pass
        else:
            raise ValueError(
                f"Navigation format must be either 'ephemeris' or 'sp3'. Got {self.profile['navigation_format']} instead."
            )

        return

    def purify(self, data: pd.DataFrame, relevant_columns: list[str]) -> pd.DataFrame:
        """Remove observations with missing data.

        Args:
            data (pd.DataFrame): DataFrame containing observational or navigation data.
            relevant_columns (list[str]): List of relevant columns to consider.

        Returns:
            pd.DataFrame: DataFrame with missing data observations removed.
        """
        # Drop duplicates columns
        data = data[~data.index.duplicated(keep="first")]
        relevant_columns = data.columns.intersection(relevant_columns)
        # Drop any rows with NA values on to_drop_na columns
        data.dropna(subset=relevant_columns, how="any", axis=0, inplace=True)
        return data

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

    @property
    def has_ionospheric_correction(self) -> bool:
        """Check if the epoch has ionospheric correction applied.

        Returns:
            bool: True if ionospheric correction is applied, False otherwise.
        """
        return self.IONOSPHERIC_KEY in self.nav_meta

    @property
    def tropospheric_model(self) -> str:
        """Get the tropospheric model used in the epoch.

        Returns:
            str: The tropospheric model used in the epoch.
        """
        return self.profile["tropospheric_model"]

    @property
    def navigation_format(self) -> str:
        """Get the navigation format used in the epoch.

        Returns:
            str: The navigation format used in the epoch.
        """
        return self.profile["navigation_format"]

    @tropospheric_model.setter
    def tropospheric_model(self, value: str) -> None:
        """Set the tropospheric model used in the epoch.

        Args:
            value (str): The value to set.
        """
        if value not in ["saastamoinen", "unb3m"]:
            raise ValueError(
                f"Tropospheric model must be either 'saastamoinen' or 'unb3m'. Got {value} instead."
            )
        self.profile["tropospheric_model"] = value

    @navigation_format.setter
    def navigation_format(self, value: str) -> None:
        """Set the navigation format used in the epoch.

        Args:
            value (str): The value to set.
        """
        if value not in ["ephemeris", "sp3"]:
            raise ValueError(
                f"Navigation format must be either 'ephemeris' or 'sp3'. Got {value} instead."
            )
        self.profile["navigation_format"] = value

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
