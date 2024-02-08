"""Data preprocessor for the triangulation algorithm.

Preprocessor Constellation Codes:
- G: GPS Epoch Data
- R: GLONASS Epoch Data
- E: Galileo Epoch Data
- J: QZSS Epoch Data
- C: BDS Epoch Data
- I: IRNSS Epoch Data
- S: SBAS Epoch Data

This module defines an abstract class, Preprocessor, which serves as a template for specific data preprocessors catering to different satellite constellations. The preprocessing involves extracting and organizing relevant information from the provided epoch data, preparing it for the subsequent triangulation algorithm.

Attributes:
    __all__ (List[str]): List of names to be exported when using "from module import *".
    Preprocessor (ABC): Abstract class for a data preprocessor.

Methods:
    __init__(self, constellation: str) -> None:
        Initializes the preprocessor with the specified constellation.

    preprocess(self, epoch: Epoch) -> tuple[pd.DataFrame, pd.DataFrame]:
        Abstract method to preprocess the given epoch data, extracting range and satellite position information.

    __repr__(self) -> str:
        Returns a string representation of the preprocessor instance.

Usage:
    from path.to.module import Preprocessor

Example:
    gps_preprocessor = Preprocessor('G')
    range_data, position_data = gps_preprocessor.preprocess(gps_epoch_data)

"""

from abc import ABC, abstractmethod

import pandas as pd

from .....epoch.epoch import Epoch

__all__ = ["Preprocessor"]


class Preprocessor(ABC):
    """This is an abstract class for a data preprocessor."""

    def __init__(self, constellation: str) -> None:
        """Initializes the preprocessor.

        Args:
            constellation (str): The constellation of the data.
        """
        self.constellation = constellation
        super().__init__()

    @abstractmethod
    def preprocess(self, epoch: Epoch, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocesses the data.

        Args:
            epoch (Epoch): The epoch to be preprocessed. Contains the observation and navigation data for each constellation.
            obs_metadata (pd.Series): Observation metadata.
            nav_metadata (pd.Series): Navigation metadata.
            kwargs: Additional keyword arguments.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: range and satellite position data for triangulation.
        """
        pass

    def __repr__(self) -> str:
        """Returns the representation of the preprocessor."""
        return f"{self.__class__.__name__}(constellation={self.constellation})"
