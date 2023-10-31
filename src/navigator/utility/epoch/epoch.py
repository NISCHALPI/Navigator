"""Represents an Epoch of Observational Data.

An Epoch is a time segment of observational data that includes a timestamp and associated observables in the form of a pandas DataFrame.

Attributes:
    timestamp (pd.Timestamp): The timestamp of the epoch.
    data (pd.DataFrame): The observational data of the epoch.

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

    def __init__(self, timestamp: pd.Timestamp, data: pd.DataFrame) -> None:
        """Initialize an Epoch instance with a timestamp and observational data.

        Args:
            timestamp (pd.Timestamp): The timestamp of the epoch.
            data (pd.DataFrame): A DataFrame containing observational data.

        """
        # Timestamp of the epoch
        self._timestamp = timestamp
        # Observational data of the epoch
        self._data = self.purify(data)

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
    def data(self) -> pd.DataFrame:
        """Get the observational data of the epoch.

        Returns:
            pd.DataFrame: A DataFrame containing observational data.

        """
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame) -> None:  # noqa: ARG002
        """Prevent direct modification of the data. Use the constructor instead.

        Args:
            data (pd.DataFrame): The data to set.

        Raises:
            AttributeError: If you try to set the data directly.

        """
        raise AttributeError("Cannot set data directly. Use the constructor instead.")

    def __repr__(self) -> str:
        """Return a string representation of the Epoch.

        Returns:
            str: A string representation of the Epoch.

        """
        return f"Epoch(timestamp={self.timestamp}, sv={self.data.shape[0]})"

    def __getitem__(self, sv: int) -> pd.Series:
        """Retrieve observables for a specific satellite vehicle (SV) by index.

        Args:
            sv (int): The index of the satellite vehicle (SV).

        Returns:
            pd.Series: A pandas Series containing observables for the specified SV.

        """
        return self.data.loc[sv]

    def __len__(self) -> int:
        """Return the number of satellite vehicles (SVs) in the epoch.

        Returns:
            int: The number of satellite vehicles (SVs) in the epoch.

        """
        return len(self.data)

    @staticmethod
    def epochify(obs: pd.DataFrame) -> list["Epoch"]:
        """Convert a pandas DataFrame of observations into a list of 'Epoch' objects.

        Parameters:
        obs (pd.DataFrame): A DataFrame containing timestamped observations.

        Returns:
        list['Epoch']: A list of 'Epoch' objects, where each 'Epoch' represents
        a unique timestamp with its associated data.

        This method takes a pandas DataFrame with timestamped data and processes it
        to create a list of 'Epoch' objects, where each 'Epoch' represents a unique
        timestamp along with the corresponding data for that timestamp. The input
        DataFrame should have a multi-index, with the second level indicating
        timestamps.
        """
        # Get the unique timestamps in the DataFrame
        timestamps = obs.index.get_level_values("time").unique()

        # Create a list of Epochs
        epoches = []

        for timestamp in timestamps:
            # Get the data for the current timestamp
            data = obs.xs(key=timestamp, level="time", drop_level=True)

            # Create an Epoch object and add it to the list
            epoches.append(Epoch(timestamp, data))

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
        
