"""Represents an Epoch of Observational Data.

An Epoch is a time segment of observational data that includes a timestamp and associated observables in the form of a pandas DataFrame.
Moreover, it also contains the navigation data for the epoch which is used for triangulation.

Attributes:
    timestamp (pd.Timestamp): The timestamp of the epoch.
    data (pd.DataFrame): The observational data of the epoch.
    nav_data (pd.DataFrame): The navigation data of the epoch.

Methods:
    __init__(timestamp, data): Initialize an Epoch instance with a timestamp and observational data.
    timestamp (property): Get the timestamp of the epoch.
    data (property): Get the observational data of the epoch.
    __getitem__(sv): Retrieve observables for a specific satellite vehicle (SV) by index.
    __repr__(): Return a string representation of the Epoch.

Args:
    timestamp (pd.Timestamp): The timestamp associated with the epoch.
    data (pd.DataFrame): A DataFrame containing observational data.

Raises:
    AttributeError: If you try to set the timestamp or data directly.

Example:
    >>> timestamp = pd.Timestamp('2023-10-12 12:00:00')
    >>> data = pd.DataFrame(...)
    >>> epoch = Epoch(timestamp, data)
    >>> print(epoch)
    Epoch(timestamp=2023-10-12 12:00:00, sv=...)

You should provide more detailed descriptions for the methods, explaining their purpose, accepted arguments, and return values. This will make the documentation more informative and help users understand how to use the class and its methods.

"""

import pickle
from pathlib import Path  # type: ignore

import pandas as pd  # type: ignore

__all__ = ["Epoch"]


class Epoch:
    """Represents an Epoch of Observational Data.

    An Epoch is a time segment of observational data that includes a timestamp and associated observables in the form of a pandas DataFrame.

    Attributes:
        timestamp (pd.Timestamp): The timestamp of the epoch.
        data (pd.DataFrame): The observational data of the epoch.
        nav_data (pd.DataFrame): The navigation data of the epoch.

    Methods:
        __init__(timestamp, data): Initialize an Epoch instance with a timestamp and observational data.
        timestamp (property): Get the timestamp of the epoch.
        data (property): Get the observational data of the epoch.
        __getitem__(sv): Retrieve observables for a specific satellite vehicle (SV) by index.
        __repr__(): Return a string representation of the Epoch.

    Args:
        timestamp (pd.Timestamp): The timestamp associated with the epoch.
        data (pd.DataFrame): A DataFrame containing observational data.

    Raises:
        AttributeError: If you try to set the timestamp or data directly.

    Example:
        >>> timestamp = pd.Timestamp('2023-10-12 12:00:00')
        >>> data = pd.DataFrame(...)  # Replace '...' with actual data.
        >>> epoch = Epoch(timestamp, data)
        >>> print(epoch)
        Epoch(timestamp=2023-10-12 12:00:00, sv=...)

    """

    def __init__(self, timestamp: pd.Timestamp, obs_data: pd.DataFrame, nav_data : pd.DataFrame, trim : bool = False) -> None:
        """Initialize an Epoch instance with a timestamp and observational data.

        Args:
            timestamp (pd.Timestamp): The timestamp of the epoch.
            obs_data (pd.DataFrame): A DataFrame containing observational data.
            nav_data (pd.DataFrame): A DataFrame containing navigation data.
            trim (bool): Intersect the satellite vehicles in the observation data and navigation data. Defaults to False.

        """
        # Timestamp of the epoch
        self._timestamp = timestamp
        
        # Observational data of the epoch
        self._obs_data = self.purify(obs_data)
        
        # Navigation data of the epoch
        self._nav_data = nav_data
        
        # Trim the data
        if trim:
            self.trim()
            
    
    def trim(self) -> None:
        """Intersect the satellite vehicles in the observation data and navigation data."""
        # Get the common satellite vehicles
        common_sv = self.obs_data.index.get_level_values("sv").intersection(
            self.nav_data.index.get_level_values("sv")
        )
        
        # Trim the data
        self._obs_data = self.obs_data.loc[self.obs_data.index.get_level_values("sv").isin(common_sv)]
        self._nav_data = self.nav_data.loc[self.nav_data.index.get_level_values("sv").isin(common_sv)]
        
        return
    
    

    def purify(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove observations with missing data."""
        # Drop NA rows values for observations ["C1C", "C2C", "C2W" , "C1W"] if present
        if "C1C" in data.columns:
            data = data.dropna(subset=["C1C"])
        if "C2C" in data.columns:
            data = data.dropna(subset=["C2C"])
        if "C2W" in data.columns:
            data = data.dropna(subset=["C2W"])
        if "C1W" in data.columns:
            data = data.dropna(subset=["C1W"])

        return data

    @property
    def timestamp(self) -> pd.Timestamp:
        """Get the timestamp of the epoch.

        Returns:
            pd.Timestamp: The timestamp associated with the epoch.

        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp: pd.Timestamp) -> None:  # noqa: ARG002
        """Prevent direct modification of the timestamp. Use the constructor instead.

        Args:
            timestamp (pd.Timestamp): The timestamp to set.

        Raises:
            AttributeError: If you try to set the timestamp directly.

        """
        raise AttributeError(
            "Cannot set timestamp directly. Use the constructor instead."
        )

    @property
    def obs_data(self) -> pd.DataFrame:
        """Get the observational data of the epoch.

        Returns:
            pd.DataFrame: A DataFrame containing observational data.

        """
        return self._obs_data

    @obs_data.setter
    def obs_data(self, data: pd.DataFrame) -> None:  # noqa: ARG002
        """Prevent direct modification of the data. Use the constructor instead.

        Args:
            data (pd.DataFrame): The data to set.

        Raises:
            AttributeError: If you try to set the data directly.

        """
        raise AttributeError("Cannot set data directly. Use the constructor instead.")
    
    @property
    def nav_data(self) -> pd.DataFrame:
        """Get the navigation data of the epoch.

        Returns:
            pd.DataFrame: A DataFrame containing navigation data.

        """
        return self._nav_data
    
    @nav_data.setter
    def nav_data(self, nav_data: pd.DataFrame) -> None:  # noqa: ARG002
        """Prevent direct modification of the nav_data. Use the constructor instead.

        Args:
            nav_data (pd.DataFrame): The nav_data to set.
        

        Raises:
            AttributeError: If you try to set the nav_data directly.
        """
        raise AttributeError("Cannot set nav_data directly. Use the constructor instead.")

    @property
    def common_sv(self) -> pd.Index:
        """Get the common satellite vehicles between the observation data and navigation data."""
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
    def epochify(obs: pd.DataFrame, nav : pd.DataFrame,  mode : str ="maxsv" ) -> list["Epoch"]:
        """Convert a pandas DataFrame of observations into a list of 'Epoch' objects.

        Parameters:
        obs (pd.DataFrame): A DataFrame containing timestamped observations.
        nav (pd.DataFrame): A DataFrame containing navigation data.
        mode (str): Method to choose the best navigation message. Defaults to 'maxsv'.

        Returns:
        list['Epoch']: A list of 'Epoch' objects, where each 'Epoch' represents
        a unique timestamp with its associated data.

        This method takes a pandas DataFrame with timestamped data and processes it
        to create a list of 'Epoch' objects, where each 'Epoch' represents a unique
        timestamp along with the corresponding data for that timestamp. The input
        DataFrame should have a multi-index, with the second level indicating
        timestamps. The navigation data is chosen based on the specified method. The navigation
        message is then intersected with the observations to create the Epoch object.
        """
        # assert that the obs and nav data are pandas DataFrames
        if not isinstance(obs, pd.DataFrame):
            raise TypeError(f"obs must be a pandas DataFrame. Got {type(obs)} instead.")
        if not isinstance(nav, pd.DataFrame):
            raise TypeError(f"nav must be a pandas DataFrame. Got {type(nav)} instead.")
        
        # Check that mode is one of 'nearest' or 'maxsv'
        if mode.lower() not in ["nearest", "maxsv"]:
            raise ValueError(
                'Invalid ephemeris method. Method must be "nearest" or "maxsv".'
            )
        
        # If any of the DataFrames are empty, then return an empty list
        if obs.empty or nav.empty:
            return []

        # Get the unique timestamps in the DataFrame
        timestamps = obs.index.get_level_values("time").unique()

        # Create a list of Epochs
        epoches = []

        for timestamp in timestamps:
            # Get the data for the current timestamp
            data = obs.xs(key=timestamp, level="time", drop_level=True)
            
            # Create an Epoch object and add it to the list
            epoches.append(Epoch._choose_nav_and_pack_obs_data(time_stamp=timestamp, obs_data=data, nav=nav, ephemeris=mode))
            
        return epoches
    
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
            raise TypeError(f"Loaded object is not an Epoch. Got {type(epoch)} instead.")
        
        return epoch
    
    
    @staticmethod
    def _choose_nav_and_pack_obs_data(
        time_stamp: pd.Timestamp,
        obs_data : pd.DataFrame , 
        nav: pd.DataFrame,
        ephemeris: str = "maxsv",
    ) -> "Epoch":
        """Choose the best navigation message based on the specified method.

        Args:
            time_stamp (pd.Timestamp): The timestamp of the epoch.
            obs_data (pd.DataFrame): The observational data of the epoch.
            nav (pd.DataFrame): Navigation data.
            ephemeris (str, optional): Method to choose the best navigation message. Defaults to 'maxsv'.

        Returns:
            tuple[Epoch,  pd.DataFrame]: The chosen navigation message and the observations with the intersected ephemeris.
        """
        # Get the epoch time
        epoch_time = time_stamp
        epoch_data = obs_data
        navtimes = nav.index.get_level_values("time").unique()

        # If the max time difference is greater that 4hours between epoch time and nav time,
        # then raise an error since ephemeris is not valid for that time
        if all(abs(navtimes - epoch_time) > pd.Timedelta("4h")):
            raise ValueError(
                f"No valid ephemeris for {epoch_time}. All ephemeris are more than 4 hours away from the epoch time."
            )

        if ephemeris.lower() == "nearest":
            # Get the nearest timestamp from epoch_time
            nearest_time = min(navtimes, key=lambda x: abs(x - epoch_time))
            ephm = nav.loc[[nearest_time]]

        elif ephemeris.lower() == "maxsv":
            # Get the timestamp with maximum number of sv
            maxsv_time = max(navtimes, key=lambda x: nav.loc[x].shape[0])
            ephm = nav.loc[[maxsv_time]]

        else:
            raise ValueError(
                'Invalid ephemeris method. Method must be "nearest" or "maxsv".'
            )

     
         # If the intersection is empty, then raise an error
        if ephm.empty:
            raise ValueError(
                "Use maxsv ephemeris mode. No common sv between observations and nav ephemeris data."
            )
        
        # Create a new obs epoch with the intersected ephemeris
        return Epoch(
            timestamp=epoch_time,
            obs_data=epoch_data,
            nav_data=ephm,
            trim=True
        )
        
