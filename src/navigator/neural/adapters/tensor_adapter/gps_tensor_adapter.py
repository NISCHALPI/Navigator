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
from ....utility.simulator.reciever_simulator import RecieverSimulator

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

    def _mask_sv(self, sv_coords: DataFrame, mask_sv: int) -> DataFrame:
        """Masks the satellite coordinates based on the number of satellites to track.

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
            stack_phase=True,
            **kwargs,
        )

        # Return the range and satellite coordinates as tensors
        return torch.from_numpy(
            pseudorange.to_numpy(dtype=np.float32)
        ), torch.from_numpy(sv_coords.to_numpy(dtype=np.float32))

    def to_tensor(
        self, epoch: Epoch, mask_sv: int = -1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the GPS epoch data to tensor data.

        Args:
            epoch: The GPS epoch data to be converted to tensor data.
            mask_sv: The number of satellites to track and train on. Defaults to -1.
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


class DummyDataset(Dataset):
    """A class for converting dummy epoch data to tensor data for NN training.

    Attributes:
        epochs: The epoch collection to be converted to a dataset.

    Raises:
        TypeError: If epochs is not an instance of EpochCollection.
    """

    # GPS tensor adapter
    tensor_adapter = GPSTensorAdatper()

    def __init__(
        self, num_points: int, simulator: RecieverSimulator, **kwargs
    ) -> None:  # noqa : ARG002
        """Initializes the DummyEpoch class.

        Args:
            num_points: The number of data points to generate.
            simulator: The simulator to generate the data points.
            **kwargs: Additional keyword arguments.

        Raises:
            TypeError: If epochs is not an instance of EpochCollection.

        """
        if not isinstance(simulator, RecieverSimulator):
            raise TypeError("simulator must be an instance of RecieverSimulator")

        # Store the number of data points
        self.num_points = num_points
        self.simulator = simulator

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return self.num_points

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the item at the given index.

        Args:
            idx: The index of the item to be returned.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   The true state, (code, sv_coords) as tensor data for KalmanNet training.
        """
        # Get the epoch data at the given index
        epoch = self.simulator.get_epoch(
            idx - self.num_points // 2
        )  # Wrap form [-num_points//2, num_points//2]

        # Preprocess the epoch data to tensor data
        code, sv_coords = self.tensor_adapter(epoch)

        # Get the true state
        true_state = torch.tensor(
            [
                epoch.real_coord["x"],
                epoch.real_coord["x_dot"],
                epoch.real_coord["y"],
                epoch.real_coord["y_dot"],
                epoch.real_coord["z"],
                epoch.real_coord["z_dot"],
                epoch.real_coord["cdt"],
            ],
            dtype=torch.float32,
        )

        # Return the true state, (code, sv_coords) as tensor data

        return true_state, code, sv_coords
