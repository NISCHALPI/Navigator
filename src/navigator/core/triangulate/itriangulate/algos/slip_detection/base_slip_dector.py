"""This module contains the base class for all cycle slip detectors."""

from abc import ABC, abstractmethod

from ......epoch import Epoch


class BaseSlipDetector(ABC):
    """Base class for all cycle slip detectors.

    The BaseSlipDetector class provides the interface for all cycle slip detectors. It defines the methods that must be implemented by all cycle slip detectors.

    Methods:
        detect(epoch): Detects cycle slips in the given epoch and returns the detected cycle slips.

    """

    L1_FREQ = 1575.42e6
    L1_WAVELENGTH = 299792458.0 / L1_FREQ
    L2_FREQ = 1227.60e6
    L2_WAVELENGTH = 299792458.0 / L2_FREQ

    def __init__(self, detector: str) -> None:
        """Constructs a BaseSlipDetector object.

        Args:
            detector (str): The name of the cycle slip detector.

        Returns:
            None

        """
        # The name of the cycle slip detector
        self.detector = detector

    @abstractmethod
    def update(self, epoch: Epoch) -> dict[str, bool]:
        """Detects cycle slips in the given epoch and returns the detected cycle slips as a dictionary for each satellite.

        Args:
            epoch (Epoch): The epoch to detect cycle slips in.

        Returns:
            dict[str, bool]: The detected cycle slips as a dictionary for each satellite.
        """
        pass

    def __repr__(self) -> str:
        """Returns the string representation of the BaseSlipDetector object.

        Returns:
            str: The string representation of the BaseSlipDetector object.

        """
        return f"{self.detector} Cycle Slip Detector"
