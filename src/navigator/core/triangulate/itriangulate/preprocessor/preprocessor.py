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

import numpy as np
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

    @staticmethod
    def to_computational_format(
        range_data: pd.DataFrame,
        sv_data: pd.DataFrame,
        sv_filter: list[str] | None = None,
        code_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Converts the preprocessed data to the format required for triangulation i.e numpy arrays.

        This method converts the preprocessed range and satellite position data which
        are by default in pandas DataFrame format to numpy arrays for the triangulation algorithm.

        The first array is of shape (2*num_sv,) and contains the code and phase range data
        and second array is of shape (2*num_sv, 3) and contains the satellite position data
        arranged to crosspond to respective code and phase range data.


        Args:
            range_data (pd.DataFrame): The preprocessed range data.
            sv_data (pd.DataFrame): The preprocessed satellite data.
            sv_filter (list[str], optional): The SVs to be included in the computation. Defaults to None i.e all SVs.
            code_only (bool, optional): Whether to include phase measurements or not. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: The range and satellite position data in numpy array format.
        """
        # If no SV filter is provided, include all SVs
        if sv_filter is None:
            sv_filter = sv_data.index

        else:
            # Check that all sv in sv_filter are present in sv_data
            if not set(sv_filter).issubset(sv_data.index):
                raise ValueError("SVs in sv_filter not present in sv_data")

        # Extract the code and phase range data
        code = range_data.loc[sv_filter, Epoch.L1_CODE_ON].to_numpy(np.float64)
        phase = range_data.loc[sv_filter, Epoch.L1_PHASE_ON].to_numpy(np.float64)
        sv_coords = sv_data.loc[sv_filter, ["x", "y", "z"]].to_numpy(np.float64)

        # If code_only is True, return only the code measurements
        if code_only:
            return code, sv_coords

        return np.hstack([code, phase]), np.vstack([sv_coords] * 2)
