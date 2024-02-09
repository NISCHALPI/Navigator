"""This module implements the Hatch filter for smoothing range measurements.

The Hatch filter is a specialized smoothing algorithm that utilizes range measurements and satellite visibility information to produce smoothed range values. It is particularly effective in mitigating noise and enhancing the accuracy of navigation data.

Classes:
    HatchFilter: Implementation of the Hatch filter for smoothing range measurements.

Attributes:
    _smoother_type (str): The type of the smoother.

Usage Example:
    ```python
    from navigation_smoothers import HatchFilter

    # Create a HatchFilter instance with a specified weight
    hatch_filter = HatchFilter(N=3.0)

    # Apply the Hatch filter to a list of navigation epochs
    smoothed_epochs = list(hatch_filter.smooth(epochs_list))
    ```

Details:
    The Hatch filter extends the HatchLikeSmoother base class and provides specific logic for smoothing range measurements. It incorporates satellite visibility maps and utilizes a weighted average to achieve robust smoothing.

Methods:
    - _error_constraint_kwargs(): Returns error constraint settings specific to the Hatch filter.
    - _calculate_update_parameters(current_epoch): Calculates smoothed range measurements and updates the satellite visibility map for the current epoch.

Attributes:
    N (float): The weight parameter used by the Hatch filter for smoothing.

Note:
    - Ensure that range measurements ('C1C' and 'L1C') are available in the epoch data before applying the Hatch filter.
    - The Hatch filter inherits from HatchLikeSmoother and inherits its smoothing logic.

See Also:
    - HatchLikeSmoother: Base class for Hatch-like smoothers.
    - Carrier-smoothing of code pseudoranges: https://gssc.esa.int/navipedia/index.php?title=Carrier-smoothing_of_code_pseudoranges

Author:
    Nischal Bhattarai
"""

from pandas import Series

from .base_smoother import HatchLikeSmoother

# CONSTANTS
L1_FREQ = 1575.42e6
L1_WAVELENGTH = 299792458 / L1_FREQ


class HatchFilter(HatchLikeSmoother):
    """Implementation of the Hatch filter for smoothing range measurements.

    The Hatch filter is a specialized smoothing algorithm that utilizes range measurements and satellite visibility information to produce smoothed range values. It is particularly effective in mitigating noise and enhancing the accuracy of navigation data.


    Methods:
        smooth(epochs): Applies the Hatch filter to a list of navigation epochs and returns the smoothed epochs.

    See Also:
        - HatchLikeSmoother: Base class for Hatch-like smoothers.
        - Carrier-smoothing of code pseudoranges: https://gssc.esa.int/navipedia/index.php?title=Carrier-smoothing_of_code_pseudoranges
    """

    def __init__(self, window: int = 100) -> None:
        """Constructs a HatchFilter object.

        Args:
            window (int): The window size for the Hatch filter. Defaults to 100.

        Note:
            The window size determines the number of observations used for smoothing. A larger window size results in a more robust smoothing effect.

        Returns:
            None
        """
        super().__init__(window=window, smoother_type="Hatch")

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

        return C1C - L1C


# Path: src/navigator/core/triangulate/itriangulate/smoothing/hatch_filter.py
