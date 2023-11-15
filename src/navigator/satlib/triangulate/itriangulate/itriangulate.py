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
        obs: Epoch,
        obs_metadata: pd.Series,
        nav_metadata: pd.Series,
        *args,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Abstract method for computing triangulated locations.

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.
            obs_metadata (pd.Series): Metadata for the observation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        

        Returns:
            pd.Series | pd.DataFrame: The computed triangulated location.
        
        Summary:
            This is an abstract method that must be implemented by subclasses. It performs the triangulation calculation using observation and navigation data.
        """
        pass

    def __call__(
        self,
        obs: Epoch,
        obs_metadata: pd.Series,
        nav_metadata: pd.Series,
        *args,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Callable method to perform triangulation.

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.
            obs_metadata (pd.Series): Metadata for the observation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        

        Returns:
            pd.Series | pd.DataFrame: The computed triangulated location.
        

        Summary:
            This method allows the instance of the `Itriangulate` class to be called like a function, and it delegates the triangulation calculation to the `_compute` method.
        """
        return self._compute(obs, obs_metadata, nav_metadata, *args, **kwargs)
    
    def __repr__(self) -> str:
        """Returns a string representation of the triangulation algorithm instance.

        Returns:
            str: A string representation of the triangulation algorithm instance.

        Summary:
            This method returns a string that represents the instance of the `Itriangulate` class, including its feature name.
        """
        return f"{self.__class__.__name__}({self.feature})"
