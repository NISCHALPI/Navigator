"""This module contains the DivergenceFreeSmoother class.

Requires the phase measurements in both L1 and L2 frequencies.
"""

from pandas import Series

from .base_smoother import HatchLikeSmoother
from ..slip_detection.base_slip_dector import BaseSlipDetector

__all__ = ["DivergenceFreeSmoother"]

# CONSTANTS
L1_FREQ = 1575.42e6
L1_WAVELENGTH = 299792458.0 / L1_FREQ
L2_FREQ = 1227.60e6
L2_WAVELENGTH = 299792458.0 / L2_FREQ


class DivergenceFreeSmoother(HatchLikeSmoother):
    """Implementation of the Divergence-Free Smoother for smoothing phase measurements.

    The Divergence-Free Smoother is a specialized smoothing algorithm that utilizes phase measurements in both L1 and L2 frequencies to produce smoothed phase values. It is particularly effective in mitigating noise and enhancing the accuracy of navigation data.

    Attributes:
        alpha (float): The weight parameter used by the Divergence-Free Smoother for smoothing.
    """

    alpha = 1 / ((L1_FREQ / L2_FREQ) ** 2 - 1)

    def __init__(self, slip_detector: BaseSlipDetector, window: int = 100) -> None:
        """Constructs a DivergenceFreeSmoother object.

        Args:
            slip_detector (BaseSlipDetector): The slip detector to be used for the Divergence-Free Smoother.
            window (int): The window size for the Divergence-Free Smoother. Defaults to 100.

        Note:
            The window size determines the number of observations used for smoothing. A larger window size results in a more robust smoothing effect.

        Returns:
            None
        """
        super().__init__(
            slip_detector=slip_detector, window=window, smoother_type="Divergence-Free"
        )

    def _current_update(self, sv_row: Series) -> float:
        """This method calculates the current update for divergence-free smoothing.

        Args:
            sv_row (Series): The observation data for the current satellite.

        Returns:
            float: The current update for the Hatch filter.
        """
        # Grab the current L1 and L2 phase measurements
        L1C = sv_row["L1C"] * L1_WAVELENGTH
        C1C = sv_row["C1C"]

        try:
            L2W = sv_row["L2W"] * L2_WAVELENGTH
        except KeyError:
            raise ValueError(
                "L2 phase measurements are required for divergence-free smoothing."
            )

        # Compute the divergence-free update
        return C1C - L1C - (2 * self.alpha * (L1C - L2W))

    def epoch_profile_update(self) -> dict[str, bool | str]:
        """This method returns the profile update for divergence-free smoothing.

        Returns:
            dict[str, Union[bool, str]]: The profile update for the current epoch according to the smoother.
        """
        return {
            "apply_iono": True,
            "apply_tropo": True,
        }


# Path: src/navigator/core/triangulate/itriangulate/smoothing/divergence_free_smoother.py
