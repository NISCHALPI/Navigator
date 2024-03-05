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

from ...core.triangulate.itriangulate.iterative.iterative_traingulation_interface import (
    IterativeTriangulationInterface,
)
from ...core.triangulate.itriangulate.preprocessor.gps_preprocessor import (
    GPSPreprocessor,
)
from ..epoch import Epoch
from ..epoch_collection import EpochCollection

__all__ = ["TensorAdapter", "GPSTensorAdatper", "GPSEpochDataset"]


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

    def __init__(self) -> None:
        """Initializes the GPSTensorAdapter class.

        Args:
            features: The features to be converted to tensor data.

        """
        super().__init__(features="GPS")

    def _mask_sv(self, sv_coords: DataFrame, mask_sv: int) -> DataFrame:
        """Masks the satellite coordinates.

        Args:
            sv_coords: The satellite coordinates to be masked.
            mask_sv: The number of satellites to track and train on.

        Returns:
            DataFrame: The masked satellite coordinates.

        """
        # If the number of satellite to track is greater than the number of available satellites
        # return the first mask_sv satellites based on the elevation
        if len(sv_coords) >= mask_sv:
            return sv_coords.nlargest(mask_sv, "elevation")

        # If the number of satellite to track is less than the number of available satellites
        # return the sv_coords padded with entries of the satellite with the highest elevation
        to_pad = mask_sv - len(sv_coords)

        # Get the satellite with the highest elevation
        max_elevation = sv_coords.nlargest(1, "elevation").iloc[0]

        # Pad the sv_coords with the satellite with the highest elevation
        for i in range(to_pad):
            sv_coords.loc[f"{max_elevation.name}_Pad{i}"] = max_elevation

        # Sort the sv_coords by elevation
        return sv_coords.sort_values("elevation", ascending=False)

    def to_tensor(
        self, epoch: Epoch, mask_sv: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the GPS epoch data to tensor data.

        Args:
            epoch: The GPS epoch data to be converted to tensor data.
            mask_sv: The number of satellites to track and train on.
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

        # Get the range and satellite coordinates
        sv_coords["range"] = pseudorange[: len(sv_coords)]
        sv_coords["phase"] = pseudorange[len(sv_coords) :]

        # Mask the satellite coordinates
        sv_coords = self._mask_sv(sv_coords, mask_sv)

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


class GPSEpochDataset(Dataset):
    """A class for creating a dataset from an epoch collection.

    Attributes:
        epochs: The epoch collection to be converted to a dataset.
        tensor_adapter: The tensor adapter to be used for converting the epoch data to tensor data.

    """

    tensor_adapter = GPSTensorAdatper()

    def __init__(
        self, epochs: EpochCollection, mask_sv: int = 5, trajectory: int = 10
    ) -> None:
        """Initializes the EpochDataset class.

        This class splits the epoch data into dataset. The trajectory is used to grab the trajectory of each
        time block of the measurement. The data shouled have the following index:

        - X: The range measurements of the satellites. (S, T , 2 * mask_sv)
        - Y : The satellite coordinates. (S ,T, mask_sv, 3)

        where S is the number of continuous pseudorange tracking blocks in collection, T is the trajectory length, mask_sv is the number of satellites to track and train on.

        Args:
            epochs: The epoch collection to be converted to a dataset.
            mask_sv: The number of satellites to track and train on.
            trajectory: The lenth of the trajectory to be used for each time block.

        """
        if not isinstance(epochs, EpochCollection):
            raise TypeError("epochs must be an instance of EpochCollection")
        if mask_sv < 1:
            raise ValueError("mask_sv must be greater than 0")

        # Set the attributes
        self.epochs = epochs
        self.mask_sv = mask_sv
        # Need a intial fix to run in phase mode
        epoch_0 = deepcopy(self.epochs[0])
        epoch_0.profile = epoch_0.INITIAL
        self.intial_fix = IterativeTriangulationInterface()(epoch_0)

        # Get the temporal blocks for the collection
        self.epochs_blocks = self.epochs.split_time_blocks(
            threashold=60
        )  # Split the collection into time blocks

        # Ensure the trajectory is not greater than any time block
        if any([len(block) < trajectory for block in self.epochs_blocks]):
            raise ValueError(
                "trajectory is greater than the length of some time blocks"
            )

        # Get the trajectory for each time block
        self.trajectory = trajectory

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return min([len(block) // self.trajectory for block in self.epochs_blocks])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Returns the item at the specified index."""
        # Get the indices for each of the time blocks
        start = idx * self.trajectory
        end = start + self.trajectory

        # Get the epoches for each of the time blocks
        epoches = [blocks[start:end] for blocks in self.epochs_blocks]

        # Tensorize the epoches
        S = len(self.epochs_blocks)
        T = self.trajectory

        # Initialize the tensors
        range_tensor = torch.zeros((S, T, 2 * self.mask_sv), dtype=torch.float32)
        coords_tensor = torch.zeros((S, T, self.mask_sv, 3), dtype=torch.float32)

        # Get the range and satellite coordinates for each time block
        for i, block in enumerate(epoches):
            for j, epoch in enumerate(block):
                range_tensor[i, j], coords_tensor[i, j] = self.tensor_adapter(
                    epoch=epoch, mask_sv=self.mask_sv, prior=self.intial_fix
                )

        return range_tensor, coords_tensor
