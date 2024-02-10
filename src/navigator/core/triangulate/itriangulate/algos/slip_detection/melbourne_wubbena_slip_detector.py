"""This module contains the Melbourne-Wubbena slip detector."""

from .base_slip_dector import BaseSlipDetector


class MelbourneWubbenaSlipDetector(BaseSlipDetector):
    """The Melbourne-Wubbena slip detector class."""

    def __init__(self):
        """Initialize the Melbourne-Wubbena slip detector."""
        super().__init__(detector="Melbourne-Wubbena")

    def _wide_laning_combination(self, l1: float, l2: float) -> float:
        """Calculate the wide laning combination of L1 and L2 carrier phase measurements and code measurements.

        Args:
            l1 (float): L1 carrier phase or code measurement.
            l2 (float): L2 carrier phase or code measurement.

        Returns:
            float: The wide laning combination of L1 and L2 carrier phase measurements and code measurements.
        """
        return (self.L1_FREQ * l1 - self.L2_FREQ * l2) / (self.L1_FREQ - self.L2_FREQ)

    # TODO : Implement the Melbourne-Wubbena slip detector
