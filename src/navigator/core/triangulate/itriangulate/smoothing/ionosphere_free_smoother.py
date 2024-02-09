"""This module contains the IonosphereFreeSmoother class.

Requires the phase measurements in both L1 and L2 frequencies and code measurements in L1 and L2 frequencies.
"""

from pandas import Series

from .base_smoother import HatchLikeSmoother

__all__ = ["IonosphereFreeSmoother"]

# CONSTANTS
L1_FREQ = 1575.42e6
L1_WAVELENGTH = 299792458.0 / L1_FREQ
L2_FREQ = 1227.60e6
L2_WAVELENGTH = 299792458.0 / L2_FREQ


class IonosphereFreeSmoother(HatchLikeSmoother):
    """Implementation of the Ionosphere-Free Smoother for smoothing phase measurements.

    The Ionosphere-Free Smoother is a specialized smoothing algorithm that utilizes phase measurements in both L1 and L2 frequencies and code measurements in L1 and L2 frequencies to produce smoothed phase values. It is particularly effective in mitigating noise and enhancing the accuracy of navigation data.

    Args:
        N (float): The weight of the smoother.


    Attributes:
        N (float): The weight parameter used by the Ionosphere-Free Smoother for smoothing.

    """

    def __init__(self, window: int = 100) -> None:
        """Constructs a IonosphereFreeSmoother object.

        Args:
            window (int): The window size for the Divergence-Free Smoother. Defaults to 100.

        Note:
            The window size determines the number of observations used for smoothing. A larger window size results in a more robust smoothing effect.

        Returns:
            None
        """
        super().__init__(window=window, smoother_type="IonosphereFree")

    def _ion_free_combination(self, l1: float, l2: float) -> float:
        """This method calculates the ionosphere-free combination of the phase measurements.

        Args:
            l1 (float): The phase measurement in L1 frequency.
            l2 (float): The phase measurement in L2 frequency.

        Returns:
            float: The ionosphere-free combination of the phase measurements.
        """
        return (L1_FREQ**2 * l1 - L2_FREQ**2 * l2) / (L1_FREQ**2 - L2_FREQ**2)

    def _current_update(self, sv_row: Series) -> float:
        """This method calculates the current update for the Hatch filter.

        Args:
            sv_row (Series): The observation data for the current satellite.

        Returns:
            float: The current update for the Hatch filter.
        """
        # Grab the current C1C and L1C measurements
        C1C = sv_row["C1C"]
        L1C = sv_row["L1C"] * L1_WAVELENGTH
        try:
            L2W = sv_row["L2W"] * L2_WAVELENGTH
            C2W = sv_row["C2W"]
        except KeyError:
            raise ValueError(
                "The phase measurements in both L1 and L2 frequencies are required for the Ionosphere-Free Smoother."
            )

        # Calculate the ionosphere-free combination
        C_IF = self._ion_free_combination(C1C, C2W)
        L_IF = self._ion_free_combination(L1C, L2W)
        # The ionosphere-free combination of the phase measurements
        return C_IF - L_IF


# Path: src/navigator/core/triangulate/itriangulate/smoothing/ionosphere_free_smoother.py
