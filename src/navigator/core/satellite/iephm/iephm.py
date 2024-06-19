"""Interface for satellite location calculation using ephemeris data.

This abstract class defines an interface for calculating the location of satellites using ephemeris data. Ephemeris data provides
information about the positions of celestial objects over time, and this interface allows different implementations
to compute satellite positions based on this data.

Attributes:
    _feature (str): A string indicating the type of satellite or feature for which ephemeris data is being used.

Methods:
    _compute(metadata: pd.Series, data: pd.DataFrame) -> pd.Series:
        Abstract method to compute satellite position based on ephemeris data.

Args:
            metadata (pd.Series): Metadata related to the ephemeris data.
            data (pd.Series): Series containing ephemeris data.

Returns:
            pd.Series: A Pandas Series representing the calculated satellite position.

    __call__(metadata: pd.Series, data: pd.DataFrame) -> pd.Series:
        Callable method that invokes the _compute method to compute satellite position.

Args:
            metadata (pd.Series): Metadata related to the ephemeris data.
            data (pd.Series): Series containing ephemeris data.

Returns:
            pd.Series: A Pandas Series representing the calculated satellite position.

    __repr__() -> str:
        Returns a string representation of the class instance, including the feature type.

Example Usage:
    This abstract class should be subclassed to implement specific satellite location calculation methods for different satellite types or features.
"""

from abc import ABC, abstractmethod

import pandas as pd  # type: ignore

__all__ = ["AbstractIephemeris"]


class AbstractIephemeris(ABC):
    """Interface for satellite location calculation using ephemeris data.

    This abstract class defines an interface for calculating the location of satellites using ephemeris data. Ephemeris data provides
    information about the positions of celestial objects over time, and this interface allows different implementations
    to compute satellite positions based on this data.
    """

    def __init__(self, feature: str = "NoneType") -> None:
        """Initialize a new instance of the AbstractIephemeris class.

        Args:
            feature (str): A string indicating the type of satellite or feature for which ephemeris data is being used.
        """
        self._feature = feature
        pass

    @abstractmethod
    def _compute(
        self, t: pd.Timestamp, metadata: pd.Series, data: pd.Series, **kwargs
    ) -> pd.Series:
        """Abstract method to compute satellite position based on ephemeris data.

        Args:
        t (pd.Timestamp): The SV time at which to compute the satellite position.
        metadata (pd.Series): Metadata related to the ephemeris data.
        data (pd.Series): Series containing ephemeris data.
        **kwargs: Additional keyword arguments.

        Returns:
        pd.Series: A Pandas Series representing the calculated satellite position.
        """
        pass

    def __call__(
        self, t: pd.Timestamp, metadata: pd.Series, data: pd.Series, **kwargs
    ) -> pd.Series:
        """Callable method that invokes the _compute method to compute satellite position.

        Args:
            t (pd.Timestamp): The SV time at which to compute the satellite position.
            metadata (pd.Series): Metadata related to the ephemeris data.
            data (pd.Series): Series containing ephemeris data.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: A Pandas Series representing the calculated satellite position.
        """
        return self._compute(t, metadata, data, **kwargs)

    def __repr__(self) -> str:
        """Returns a string representation of the class instance, including the feature type.

        Returns:
            str: A string representation of the class instance.
        """
        return f"{self.__class__.__name__}({self._feature})"
