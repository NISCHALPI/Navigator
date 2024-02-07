"""Itriangulate Interface for triangulation algorithms.

This module defines an abstract base class `Itriangulate` for triangulation algorithms. Triangulation algorithms are used to estimate locations based on observations and navigation data.

Attributes:
    __all__ (list): A list of names to be exported when using `from module import *`.

Methods:
    __init__(self, feature: str) -> None:
        Initializes an instance of the triangulation algorithm with a specific feature name.

    _compute(self, obs: pd.DataFrame, nav: pd.DataFrame, *args, **kwargs) -> pd.Series | pd.DataFrame:
        Abstract method to compute the triangulated location. Subclasses must implement this method.

    __call__(self, obs: pd.DataFrame, nav: pd.DataFrame, *args, **kwargs) -> pd.Series | pd.DataFrame:
        Callable method to perform triangulation. It delegates to the `_compute` method.

    __repr__(self) -> str:
        Returns a string representation of the triangulation algorithm instance.

Note:
    This is an abstract base class and should not be instantiated directly. Subclasses must implement the `_compute` method to provide triangulation functionality.
"""

from abc import ABC, abstractmethod

import pandas as pd

from ....utility import Epoch
from .preprocess.gps_preprocessor import GPSPreprocessor
from .preprocess.preprocessor import Preprocessor

__all__ = ["Itriangulate"]


class Itriangulate(ABC):
    """Itriangulate Interface for triangulation algorithms."""

    def __init__(self, feature: str) -> None:
        """Initializes an instance of the triangulation algorithm with a specific feature name.

        Args:
            feature (str): The name of the feature for which triangulation is being performed.

        Returns:
            None

        Summary:
            This method initializes an instance of the `Itriangulate` class with the specified feature name.
        """
        self.feature = feature

    @abstractmethod
    def _compute(
        self,
        epoch: Epoch,
        *args,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Abstract method for computing triangulated locations.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.


        Returns:
            pd.Series | pd.DataFrame: The computed triangulated location.

        Summary:
            This is an abstract method that must be implemented by subclasses. It performs the triangulation calculation using observation and navigation data.
        """
        pass

    def _auto_dispatch_preprocessor(self, epoch: Epoch) -> Preprocessor:
        """Auto dispatches the preprocessor based on the constellation.

        Dispatching mechanism is based on the PRN of the satellites in the epoch observation data.
        i.e if "G01" is present in the epoch observation data, then the GPS preprocessor is dispatched.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.

        Returns:
            Preprocessor: The preprocessor for the constellation.
        """
        # Get the common satellites in the epoch observation data
        prns = epoch.common_sv

        # Check that all the satellites are of the same constellation
        first_prn = prns[0][0]  # Get the first constellation code
        assert all(
            first_prn == prns[0][0] for prn in prns
        ), "Satellites PRNs are not of the same constellation"

        # Dispatch the preprocessor based on the constellation
        if first_prn == "G":
            return GPSPreprocessor()

        # If the constellation is not supported, raise an error
        raise ValueError(f"Invalid constellation: {first_prn}")

    def _preprocess(self, epoch: Epoch, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocesses the data.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            **kwargs: Additional keyword arguments passed to the preprocessor.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: range and satellite position dataframes.
        """
        # Auto dispatch the preprocessor
        preprocesser = self._auto_dispatch_preprocessor(epoch)

        # Preprocess the data
        return preprocesser.preprocess(epoch, **kwargs)

    def __call__(
        self,
        obs: Epoch,
        *args,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Callable method to perform triangulation.

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.


        Returns:
            pd.Series | pd.DataFrame: The computed triangulated location.


        Summary:
            This method allows the instance of the `Itriangulate` class to be called like a function, and it delegates the triangulation calculation to the `_compute` method.
        """
        return self._compute(obs, *args, **kwargs)

    def __repr__(self) -> str:
        """Returns a string representation of the triangulation algorithm instance.

        Returns:
            str: A string representation of the triangulation algorithm instance.

        Summary:
            This method returns a string that represents the instance of the `Itriangulate` class, including its feature name.
        """
        return f"{self.__class__.__name__}({self.feature})"
