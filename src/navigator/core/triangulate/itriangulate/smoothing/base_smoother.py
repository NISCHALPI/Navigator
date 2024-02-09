"""This module serves as the foundation for implementing various smoothing algorithms for navigation data epochs.

It defines an abstract base class, BaseSmoother, which outlines the common interface and structure for all smoothers. Smoothing is a crucial step in processing navigation data, providing improved accuracy and mitigating noise and outliers.

Classes:
    BaseSmoother (ABC): The abstract base class for all smoothers. It enforces a consistent structure and interface for smoother implementations.

Attributes:
    SMOOOTHING_KEY (str): A constant representing the key used for storing smoothed range data within navigation epochs.

Usage Example:
    ```python
    from navigation_smoothers import BaseSmoother

    class MySmoother(BaseSmoother):
        # Custom implementation of smoothing logic
        def smooth(self, epochs):
            # ... implementation here ...
    ```

See Also:
    - Carrier Smoothing: [Carrier-smoothing of code pseudoranges](https://gssc.esa.int/navipedia/index.php?title=Carrier-smoothing_of_code_pseudoranges)
    - Code-Carrier Divergence: [Code-Carrier Divergence Effect](https://gssc.esa.int/navipedia/index.php?title=Code-Carrier_Divergence_Effect)

Note:
    Implementing smoothers based on this module requires overriding the abstract methods defined in BaseSmoother, ensuring a consistent interface.

For more detailed information and usage guidelines, refer to the provided external resources.

Author:
    Nischal Bhattarai

"""

from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd

from .....epoch import Epoch

__all__ = ["BaseSmoother"]

# CONSTANTS
L1_FREQ = 1575.42e6
L1_WAVELENGTH = 299792458.0 / L1_FREQ


class BaseSmoother(ABC):
    """The base class for all smoothers. It is an abstract class that defines the interface for all smoothers.

    Attributes:
        _smoother_type (str): The type of the smoother.
    """

    SMOOOTHING_KEY = "SMOOTHED_RANGE"

    def __init__(self, smoother_type: str) -> None:
        """Constructs a BaseSmoother object.

        Args:
            smoother_type (str): The type of the smoother.
        """
        self._smoother_type = smoother_type

    @abstractmethod
    def smooth(self, epoches: list[Epoch]) -> list[Epoch]:
        """This method returns an iterator of smoothed epochs.

        Args:
            epoches (list[Epoch]): The epochs to be smoothed.

        Returns:
            list[Epoch]: A list of smoothed epochs.
        """
        pass

    def __repr__(self) -> str:
        """Returns the string representation of the BaseSmoother object.

        Returns:
            str: The string representation of the BaseSmoother object.
        """
        return f"{self._smoother_type}Smoother()[epoches={len(self._epoches)}]"


class HatchLikeSmoother(BaseSmoother):
    """This class implements the Hatch-like smoother.

    Attributes:
        _smoother_type (str): The type of the smoother.
    """

    def __init__(self, window: int, smoother_type: str) -> None:
        """Constructs a HatchLikeSmoother object.

        Args:
            window (int): The window size for the smoother.
            smoother_type (str): The type of the smoother.
        """
        # The window size for the smoother
        self.window = window
        # Create a SV visibility map
        self._sv_visibility_map = {}
        super().__init__(smoother_type)

    def _update_sv_visibility_map(
        self, new_sv_parameters: dict[str, (float, int)]
    ) -> None:
        """This method updates the SV visibility map with the given SV parameters for the current epoch.

        Warning: The unvisible SVs are removed from the SV visibility map since hatch-like smoother needs continuous SV visibility.

        Args:
            new_sv_parameters (dict[str, (float, int)]): The SV parameters for the current epoch.
        """
        # Update the SV visibility map
        self._sv_visibility_map.update(new_sv_parameters)

        # # Remove the unvisible SVs from the SV visibility map
        # for curr_sv in new_sv_parameters.keys():
        #     if curr_sv not in self._sv_visibility_map:
        #         self._sv_visibility_map.pop(curr_sv)

        return

    def _smoothing_logic(self, current_epoch: Epoch) -> Epoch:
        """This method implements the smoothing logic for the Hatch-like smoother.

        Args:
            current_epoch (Epoch): The current epoch to be smoothed.

        Returns:
            tuple[Epoch, dict[str, float]]: The smoothed epoch and the SV visibility map for the current epoch.
        """
        # Calculate the update parameters for the current epoch
        smoothed_parameters, sv_visibility_map = self._update(current_epoch)

        # Add the smoothed range to the current epoch
        current_epoch.obs_data[self.SMOOOTHING_KEY] = smoothed_parameters

        # Update the SV visibility map
        self._update_sv_visibility_map(sv_visibility_map)

        # Attach Smooth flag to the epoch
        current_epoch.is_smoothed = True

        return current_epoch

    def _update(
        self, current_epoch: Epoch
    ) -> tuple[pd.Series, dict[str, (float, int)]]:
        """This method calculates the smoothed range for the current epoch and updates the SV visibility map.

        Note: Need to be implemented by the derived classes.

        Args:
            current_epoch (Epoch): The current epoch to be smoothed.

        Returns:
            tuple[pd.Series, dict[str, (float, int)]]: The smoothed range for the current epoch and the SV visibility map to be used for the next epoch.
        """
        # Initialize the parameters
        prev_avg_parameter = {}
        smoothed_range = {}
        new_avg_parameter = {}

        # Must have at least C1C and L1C
        if (
            "C1C" not in current_epoch.obs_data.columns
            or "L1C" not in current_epoch.obs_data.columns
        ):
            raise ValueError("The range measurements are not available for smoothing.")

        for sv, row in current_epoch.obs_data.iterrows():
            # Grab the measurements
            L1C = row["L1C"] * L1_WAVELENGTH

            # Hatch Like Update
            curr_update = self._current_update(row)
            if sv in self._sv_visibility_map:
                prev_avg_parameter[sv] = self._sv_visibility_map[sv]
                # Check if the window size is reached
                if prev_avg_parameter[sv][1] > self.window:
                    prev_avg_parameter[sv] = (prev_avg_parameter[sv][0], self.window)
            else:
                prev_avg_parameter[sv] = (curr_update, 1.0)

            # Calculate the current count of updates
            N = prev_avg_parameter[sv][1]

            new_avg_parameter[sv] = (
                (1 / N) * curr_update + ((N - 1) / N) * prev_avg_parameter[sv][0],
                N + 1,  # Add 1 to the number of updates
            )

            # Calculate the smoothed range
            smoothed_range[sv] = L1C + new_avg_parameter[sv][0]

        return pd.Series(smoothed_range), new_avg_parameter

    @abstractmethod
    def _current_update(self, sv_row: pd.Series) -> float:
        """This method calculates the smoothed range for the current epoch and updates the SV visibility map.

        Note: The logic for calculating the current update must be implemented by the derived classes.

        Args:
            sv_row (pd.Series): The observation data for the current satellite.

        Returns:
            tuple[pd.Series, dict[str, (float, int)]]: The smoothed range for the current epoch and the SV visibility map to be used for the next epoch.
        """
        pass

    def smooth(self, epoches: list[Epoch]) -> list[Epoch]:
        """This method returns an iterator of smoothed epochs.

        Args:
            epoches (list[Epoch]): The epochs to be smoothed.

        Returns:
            Epoch: A list of smoothed epochs.
        """
        # Calculate the smoothed epochs
        smoothed = [self._smoothing_logic(deepcopy(epoch)) for epoch in epoches]

        # Reset the sv visibility map for the next smoothing
        self._sv_visibility_map.clear()
        return smoothed

    @property
    def window(self) -> int:
        """The window size for the smoother."""
        return self._window

    @window.setter
    def window(self, window: int) -> int:
        """Sets the window size for the smoother."""
        if window <= 1:
            raise ValueError("The window size must be greater than 1.")
        self._window = window
        return


# Path: src/navigator/core/triangulate/itriangulate/smoothing/base_smoother.py
