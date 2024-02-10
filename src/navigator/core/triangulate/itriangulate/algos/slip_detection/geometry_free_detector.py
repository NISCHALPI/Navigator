"""This module contains the geometry-free cycle slip detector.

Source:
    https://gssc.esa.int/navipedia/index.php?title=Detector_based_in_carrier_phase_data:_The_geometry-free_combination

The GeometryFreeDetector class in this module provides a cycle slip detection mechanism based on the geometry-free combination of carrier phase data. The algorithm is designed to identify cycle slips by comparing observed values with predictions using polynomial interpolation.

Usage:
    from [module_path] import GeometryFreeDetector

Example:
    detector = GeometryFreeDetector(window=10, order=3)
    cycle_slips = detector.update(epoch_data, sampling_rate=30)

Classes:
    GeometryFreeDetector: A class for detecting geometry-free cycle slips in carrier phase data.

Attributes:
    L1_FREQ (float): Frequency of L1 signal (1575.42e6 Hz).
    L1_WAVELENGTH (float): Wavelength of L1 signal.
    L2_FREQ (float): Frequency of L2 signal (1227.60e6 Hz).
    L2_WAVELENGTH (float): Wavelength of L2 signal.

"""

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from ......epoch import Epoch
from .base_slip_dector import BaseSlipDetector


class GeometryFreeDetector(BaseSlipDetector):
    """A class for detecting geometry-free cycle slips in carrier phase data.

    The GeometryFreeDetector class provides a cycle slip detection mechanism based on the geometry-free combination of carrier phase data. The algorithm is designed to identify cycle slips by comparing observed values with predictions using polynomial interpolation.

    Attributes:
        T_0 (float): The maximum value of the threashold.
        a_0 (float): The minimum value of the threashold.
        window (int): The window size for the filter.
        order (int): The order of the filter.
        detector_map (dict[str, list[float]]): A dictionary containing the geometry free combination for each satellite.

    """

    # Default values
    T_0 = 60
    a_0 = (3 / 2) * (BaseSlipDetector.L2_WAVELENGTH - BaseSlipDetector.L1_WAVELENGTH)

    def __init__(self, window: int = 10, order: int = 3) -> float:
        """Constructs a GeometryFreeDetector object.

        Args:
            a_0 (float): The maximum value of the threashold.
            a_1 (float): The minimum value of the threashold.
            window (int): The window size for the filter.
            order (int): The order of the filter.

        Returns:
            None

        """
        # Set the order
        if order <= 0:
            raise ValueError("The order must be a positive number.")
        self.order = order

        # Set the window size
        if window < self.order + 1:
            raise ValueError(
                "The window size must be one greater than the order of the filter."
            )
        self.window = window
        # Set up the geometry free queue
        self.detector_map = {}

        # Call the super class constructor
        super().__init__(detector="GeometryFree")

    def threashold(self, sampling_rate: float) -> float:
        """Calculates the threashold for the detector.

        Returns:
            sampling_rate (float): The sampling rate of the observations.
            float: The threashold for the detector.

        """
        return self.a_0 * (1 - np.exp(-sampling_rate / self.T_0) / 2)

    @staticmethod
    def _check(current: float, past_val: list, order: int, threashold: float) -> bool:
        """Interpolates the values in the given queue.

        Args:
            current (float): The current value to check.
            past_val (list): The past values to interpolate.
            order (int): The order of the filter.
            threashold (float): The threashold for the detector.


        Returns:
            bool : checks cycle slips if predicted is greater than threashold.
        """
        # If the length of the queue is less than the order, return False
        if len(past_val) < order + 1:
            return False

        # Fit the polynomial to the values in the queue
        coeff = Polynomial.fit(x=range(len(past_val)), y=past_val, deg=order)

        # Predict the next value
        predicted = coeff(len(past_val))

        # Check if the predicted value is greater than the threashold
        if abs(current - predicted) > threashold:
            return True
        return False

    def update(self, epoch: Epoch, sampling_rate: float = 30) -> dict[str, bool]:
        """Detects cycle slips in the given epoch and returns the detected cycle slips as a dictionary for each satellite.

        Args:
            epoch (Epoch): The epoch to detect cycle slips in.
            sampling_rate (float): The sampling rate of the observations.

        Returns:
            dict[str, bool]: The detected cycle slips as a dictionary for each satellite.

        """
        # Get the geometry free combination
        if "L1C" not in epoch.obs_data or "L2W" not in epoch.obs_data:
            raise ValueError(
                "The epoch does not contain ovservations for the L1C and L2W frequencies."
            )
        # Calculate the geometry free combination for each satellite
        geometry_free = (
            epoch.obs_data["L1C"] * BaseSlipDetector.L1_WAVELENGTH
            - epoch.obs_data["L2W"] * BaseSlipDetector.L2_WAVELENGTH
        )

        # # Remove the satellites that are not in the geometry free combination
        # # from the detector map
        # self._remove_invisible_satellites(geometry_free)

        # Detect cycle slips
        cycle_slips = {
            satellite: self._check(
                geometry_free[satellite],
                self.detector_map.get(satellite, []),
                self.order,
                self.threashold(sampling_rate),
            )
            for satellite in geometry_free.keys()
        }

        # If cycle slips are detected, clear the detector map
        # of the cycle slip satellites
        for satellite, detected in cycle_slips.items():
            if detected:
                self.detector_map.pop(satellite, None)

        # Update the detector map
        self._update_detector_map(geometry_free)

        return cycle_slips

    def _remove_invisible_satellites(self, geometry_free: pd.Series) -> None:
        """Removes the satellites that are not in the geometry free combination from the detector map.

        Args:
            geometry_free (pd.Series): The geometry free combination for each satellite.

        Returns:
            None
        """
        # Remove the satellites that are not in the geometry free combination
        # in current epoch but are in the detector map
        dmap_keys = list(self.detector_map.keys())
        for satellite in dmap_keys:
            if satellite not in geometry_free:
                self.detector_map.pop(satellite)

        return

    def _update_detector_map(self, geometry_free: pd.Series) -> None:
        """Updates the detector map with the given geometry free combination.

        Args:
            geometry_free (pd.Series): The geometry free combination for each satellite.

        Returns:
            None
        """
        # Update the detector map
        for satellite, value in geometry_free.items():
            # If the satellite is not in the detector map, add it
            if satellite not in self.detector_map:
                self.detector_map[satellite] = [value]

            else:
                # If the length of the queue is greater than the window size, remove the first value
                if len(self.detector_map[satellite]) > self.window:
                    self.detector_map[satellite].pop(0)

                # Append the value to the queue
                self.detector_map[satellite].append(value)
        return

    def reset(self) -> None:
        """Resets the detector map.

        Returns:
            None
        """
        self.detector_map = {}
        return
