"""This module contains the tensor adapter utility for converting epoch data to tensor data.

Classes:
    TensorAdapter: A class for converting epoch data to tensor data for NN training.
    GPSTensorAdapter: A class for converting GPS epoch data to tensor data for NN training.

"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Iterator

import numpy as np
import torch
from pandas.core.api import DataFrame
from torch.utils.data import Dataset

from ....core.triangulate.itriangulate.preprocessor.dummy_preprocessor import (
    DummyPreprocessor,
)
from ....core.triangulate.itriangulate.preprocessor.gps_preprocessor import (
    GPSPreprocessor,
)
from ....epoch.epoch import Epoch

__all__ = ["TensorAdapter", "GPSTensorAdatper", "DummyDataset"]


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
    def to_tensor(self, epoch: Epoch, **kwargs) -> tuple[torch.Tensor, ...]:
        """Converts the epoch data to tensor data.

        Args:
            epoch: The epoch data to be converted to tensor data.
            **kwargs: Additional keyword arguments.

        Returns:
            The epoch data as tensor data.

        """
        pass

    def __call__(self, epoch: Epoch, **kwargs) -> tuple[torch.Tensor, ...]:
        """Converts the epoch data to tensor data.

        Args:
            epoch: The epoch data to be converted to tensor data.
            **kwargs: Additional keyword arguments.

        Returns:
            The epoch data as tensor data.

        """
        return self.to_tensor(epoch, **kwargs)

    def __repr__(self) -> str:
        """String representation of the TensorAdapter class."""
        return f"{self.__class__.__name__}(features={self.features})"

    def to_tensor_bulk(
        self, epochs: list[Epoch], **kwargs
    ) -> Iterator[tuple[torch.Tensor, ...]]:
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

    # Dummy preprocessor for simulated data
    dummy_preprocessor = DummyPreprocessor()

    def __init__(self) -> None:
        """Initializes the GPSTensorAdapter class.

        Args:
            features: The features to be converted to tensor data.

        """
        super().__init__(features="GPS")

    def _dummy_tensor_adapter(
        self, epoch: Epoch, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tensor adapter for simulated dummy epoch data.

        Args:
            epoch: The GPS epoch data to be converted to tensor data.
            kwargs: Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The code, phase and sv_coords as tensor data.
        """
        # Get the range and satellite coordinates from dummy preprocessor
        pseudorange, sv_coords = self.dummy_preprocessor.preprocess(
            epoch=epoch,
            **kwargs,
        )

        return torch.from_numpy(
            pseudorange.to_numpy(dtype=np.float32)
        ), torch.from_numpy(sv_coords.to_numpy(dtype=np.float32))

    def to_tensor(self, epoch: Epoch, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the GPS epoch data to tensor data.

        Args:
            epoch: The GPS epoch data to be converted to tensor data.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The code, phase and sv_coords as tensor data.

        """
        # Copy the epoch data to avoid modifying the original data
        epoch = deepcopy(epoch)

        # If dummy profile, dispatch to dummy tensor adapter
        if epoch.profile["mode"] == "dummy":
            return self._dummy_tensor_adapter(epoch, **kwargs)

        # Set epoch profile to initial
        # Note: This ensure nothing prior to the current epoch is used
        epoch.profile = epoch.PHASE

        # Get the range and satellite coordinates
        pseudorange, sv_coords = self.preprocessor.preprocess(epoch=epoch, **kwargs)

        # Get the range and satellite coordinates
        sv_coords["range"] = pseudorange[: len(sv_coords)]
        sv_coords["phase"] = pseudorange[len(sv_coords) :]

        # Stack the range and phase [range, phase] to from measurement
        pseudorange = np.hstack(
            (
                sv_coords["range"].to_numpy(dtype=np.float32),
                sv_coords["phase"].to_numpy(dtype=np.float32),
            ),
            dtype=np.float32,
        )
        sv_coords = sv_coords[["x", "y", "z"]].to_numpy(dtype=np.float32)

        # Convert the range and coords to tensors
        return torch.from_numpy(pseudorange), torch.from_numpy(sv_coords)

    def to_tensor_bulk(
        self, epochs: list[Epoch], mask_sv: int = -1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the GPS epoch data to tensor data in bulk.

        Args:
            epochs: The GPS epoch data to be converted to tensor data.
            mask_sv: The number of satellites to track and train on. Defaults to -1.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The code, phase and sv_coords as tensor data.

        """
        data = [self.to_tensor(epoch, mask_sv=mask_sv, **kwargs) for epoch in epochs]
        range_data, sv_coords_data = zip(*data)
        return torch.stack(range_data), torch.stack(sv_coords_data)
