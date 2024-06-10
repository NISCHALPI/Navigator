"""Implements the preprocessor for the simulated epoch data."""

import pandas as pd
from pandas.core.api import DataFrame, Series

from .....epoch.epoch import Epoch
from ..algos.combinations.range_combinations import (
    ionosphere_free_combination,
)
from .preprocessor import Preprocessor

__all__ = ["DummyPreprocessor"]


class DummyPreprocessor(Preprocessor):
    """Implements the preprocessor for the simulated epoch data."""

    def __init__(self) -> None:
        """Initialize the preprocessor."""
        super().__init__(constellation="SimGPS")

    def preprocess(
        self, epoch: Epoch, **kwargs  # noqa : ARG003
    ) -> tuple[Series, DataFrame]:
        """Preprocess the epoch data.

        Args:
            epoch (Epoch): The epoch data to preprocess.
            kwargs: Additional keyword arguments.

        Returns:
            tuple[Series, DataFrame]: The preprocessed pseudorange and satellite data.
        """
        # Get the ionosphere-free combination
        code_ion_free = ionosphere_free_combination(
            p1=epoch.obs_data[epoch.L1_CODE_ON].to_numpy(),
            p2=epoch.obs_data[epoch.L2_CODE_ON].to_numpy(),
        )
        # Convert to series
        code_ion_free = Series(
            code_ion_free, index=epoch.obs_data.index, name=Epoch.L1_CODE_ON
        )

        # Get the ionosphere-free combination of the phase data
        phase_ion_free = ionosphere_free_combination(
            p1=epoch.obs_data[epoch.L1_PHASE_ON].to_numpy(),
            p2=epoch.obs_data[epoch.L2_PHASE_ON].to_numpy(),
        )

        # Convert to series
        phase_ion_free = Series(
            phase_ion_free, index=epoch.obs_data.index, name=Epoch.L1_PHASE_ON
        )

        # Concatenate the series [code_ion_free, phase_ion_free]
        code_ion_free = pd.concat([code_ion_free, phase_ion_free], axis=1)

        # Return the preprocessed data
        return (
            code_ion_free,
            epoch.nav_data[["x", "y", "z"]],
        )
