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
        self, epoch: Epoch, **kwargs
    ) -> tuple[Series, DataFrame]:  # noqa : ARG003
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
        code_ion_free = Series(code_ion_free, index=epoch.obs_data.index)

        if kwargs.get("stack_phase", False):
            # Get the ionosphere-free combination of the phase data
            phase_ion_free = ionosphere_free_combination(
                p1=epoch.obs_data[epoch.L1_PHASE_ON].to_numpy(),
                p2=epoch.obs_data[epoch.L2_PHASE_ON].to_numpy(),
            )
            # Rename the index to match the code_ion_free with Phase suffix
            additional_idx = [f"{prn}_L" for prn in epoch.obs_data.index]

            # Convert to series
            phase_ion_free = Series(phase_ion_free, index=additional_idx)

            # Concatenate the series [code_ion_free, phase_ion_free]
            code_ion_free = pd.concat([code_ion_free, phase_ion_free], axis=0)

        # Return the preprocessed data
        return (
            code_ion_free,
            epoch.nav_data[["x", "y", "z"]],
        )
