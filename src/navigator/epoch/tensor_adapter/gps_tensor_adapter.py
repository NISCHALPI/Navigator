"""This module contains the tensor adapter utility for converting epoch data to tensor data.

Classes:
    TensorAdapter: A class for converting epoch data to tensor data for NN training.
    GPSTensorAdapter: A class for converting GPS epoch data to tensor data for NN training.

"""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch
from typing import Iterator

from ...core.triangulate.itriangulate.algos.combinations import (
    ionosphere_free_combination,
)
from ...core.triangulate.itriangulate.preprocessor.gps_preprocessor import (
    GPSPreprocessor,
)
from ..epoch import Epoch

# CONSTANTS
# GPS L1 and L2 frequencies
L1_FREQ = 1575.42e6
L2_FREQ = 1227.60e6

# Calculate L1 and L2 wavelengths
L1_WAVELENGTH = 299792458 / L1_FREQ
L2_WAVELENGTH = 299792458 / L2_FREQ

__all__ = ["TensorAdapter", "GPSTensorAdatper"]


class TensorAdapter(ABC):
    """A class for converting epoch data to tensor data for NN training.

    Attributes:
        data: The epoch data to be converted to tensor data..
    """

    def __init__(self, features: str) -> None:
        """Initializes the TensorAdapter class.

        Args:
            features: The features to be converted to tensor data.

        """
        self.features = features

    @abstractmethod
    def to_tensor(self, epoch: Epoch, **kwargs) -> torch.Tensor:
        """Converts the epoch data to tensor data.

        Args:
            epoch: The epoch data to be converted to tensor data.

        Returns:
            The epoch data as tensor data.

        """
        pass

    def __call__(self, epoch: Epoch, **kwargs) -> torch.Tensor:
        """Converts the epoch data to tensor data.

        Args:
            epoch: The epoch data to be converted to tensor data.

        Returns:
            The epoch data as tensor data.

        """
        return self.to_tensor(epoch, **kwargs)

    def __repr__(self) -> str:
        """String representation of the TensorAdapter class."""
        return f"{self.__class__.__name__}(features={self.features})"

    def to_tensor_bulk(
        self, epochs: list[Epoch], **kwargs
    ) -> Iterator[tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]:
        """Converts the epoch data to tensor data in bulk.

        Args:
            epochs: The epoch data to be converted to tensor data.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[list[torch.Tensor]]: The epoch data as tensor data.

        """
        yield from (self.to_tensor(epoch, **kwargs) for epoch in epochs)


class GPSTensorAdatper(TensorAdapter):
    """A class for converting GPS epoch data to tensor data for NN training.

    Attributes:
        data: The GPS epoch data to be converted to tensor data.
    """

    # Preprocessor to calculate the sv_coords and pseudoranges
    preprocessor = GPSPreprocessor()

    # Phase and Code measurements
    L1_CODE_ON = "C1C"
    L2_CODE_ON = "C2W"
    L1_PHASE_ON = "L1C"
    L2_PHASE_ON = "L2W"

    def __init__(self) -> None:
        """Initializes the GPSTensorAdapter class.

        Args:
            features: The features to be converted to tensor data.

        """
        super().__init__(features="GPS")

    def to_tensor(self, epoch: Epoch, **kwargs) -> torch.Tensor:
        """Converts the GPS epoch data to tensor data.

        Args:
            epoch: The GPS epoch data to be converted to tensor data.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The code, phase and sv_coords as tensor data.

        """
        # Copy the epoch data to avoid modifying the original data
        epoch = deepcopy(epoch)
        # Set epoch profile to initial
        # Note: This ensure nothing prior to the current epoch is used
        epoch.profile = epoch.INITIAL

        # Preprocess the GPS data to get the sv_coords and pseudoranges
        _, sv_coords = self.preprocessor.preprocess(epoch, **kwargs)

        # Ignore the pseudoranges
        # Instead use code and phase measurements on L1 and L2 frequencies
        c1c = epoch.obs_data[self.L1_CODE_ON]
        c2w = epoch.obs_data[self.L2_CODE_ON]
        l1c = (
            epoch.obs_data[self.L1_PHASE_ON] * L1_WAVELENGTH
        )  # Ensure the phase is in meters
        l2w = (
            epoch.obs_data[self.L2_PHASE_ON] * L2_WAVELENGTH
        )  # Ensure the phase is in meters
        # Convet to numpy array
        c1c = np.array(c1c, dtype=np.float64)
        c2w = np.array(c2w, dtype=np.float64)
        l1c = np.array(l1c, dtype=np.float64)
        l2w = np.array(l2w, dtype=np.float64)

        # Calculate the ionosphere-free combination
        icode = torch.from_numpy(ionosphere_free_combination(c1c, c2w))
        iphase = torch.from_numpy(ionosphere_free_combination(l1c, l2w))

        # Return the tensor data
        return icode, iphase, torch.from_numpy(sv_coords.to_numpy(dtype=np.float64))
