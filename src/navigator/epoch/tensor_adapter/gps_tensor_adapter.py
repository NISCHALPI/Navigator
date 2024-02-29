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
from torch.utils.data import Dataset

from ...core.triangulate.itriangulate.iterative.iterative_traingulation_interface import (
    IterativeTriangulationInterface,
)
from ...core.triangulate.itriangulate.preprocessor.gps_preprocessor import (
    GPSPreprocessor,
)
from ..epoch import Epoch
from ..epoch_collection import EpochCollection

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
            **kwargs: Additional keyword arguments.

        Returns:
            The epoch data as tensor data.

        """
        pass

    def __call__(self, epoch: Epoch, **kwargs) -> torch.Tensor:
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
        epoch.profile = epoch.PHASE

        # Get the range and satellite coordinates
        pseudorange, sv_coords = self.preprocessor.preprocess(epoch=epoch, **kwargs)

        # Convert the range and coords to tensors
        return torch.from_numpy(
            pseudorange.to_numpy(dtype=np.float32)
        ), torch.from_numpy(sv_coords[["x", "y", "z"]].to_numpy(dtype=np.float32))


class EpochDataset(Dataset):
    """A class for creating a dataset from an epoch collection.

    Attributes:
        epochs: The epoch collection to be converted to a dataset.
        tensor_adapter: The tensor adapter to be used for converting the epoch data to tensor data.

    """

    def __init__(self, epochs: EpochCollection, tensor_adapter: TensorAdapter) -> None:
        """Initializes the EpochDataset class.

        Args:
            epochs: The epoch collection to be converted to a dataset.
            tensor_adapter: The tensor adapter to be used for converting the epoch data to tensor data.

        """
        if not isinstance(epochs, EpochCollection):
            raise TypeError("epochs must be an instance of EpochCollection")

        if not isinstance(tensor_adapter, TensorAdapter):
            raise TypeError("tensor_adapter must be an instance of TensorAdapter")

        # Set the attributes
        self.epochs = epochs
        self.tensor_adapter = tensor_adapter

        # Need a intial fix to run in pahse mode
        epoch_0 = deepcopy(self.epochs[0])
        epoch_0.profile = epoch_0.INITIAL
        self.intial_fix = IterativeTriangulationInterface()(epoch_0)

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.epochs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the item at the specified index."""
        return self.tensor_adapter(self.epochs[idx], prior=self.intial_fix)
