"""Dual Frequency Corrections for C/A Code.

This module provides a function for calculating dual frequency corrections for the C/A (Coarse Acquisition) code used in satellite navigation. Dual frequency corrections are applied based on the L1 and L2 ranges to improve accuracy.

Constants:
    c (float): Speed of light in meters per second.

Functions:
    dual_channel_correction(l1_range: float, l2_range: float) -> float:
        Calculate the dual channel correction value for a given L1 and L2 range.

Args:
    l1_range (float): The L1 range in meters.
    l2_range (float): The L2 range in meters.

Returns:
    float: The dual channel correction value to be applied to improve accuracy.

Summary:
    This module provides functionality to calculate dual frequency corrections for the C/A code used in satellite navigation. The `dual_channel_correction` function takes L1 and L2 ranges as input and returns the correction value to improve the accuracy of navigation calculations.
"""

# Constants
c = 299792458.0  # Speed of light in m/s


def dual_channel_correction(l1_range: float, l2_range: float) -> float:
    """Calculate the dual channel correction value for a given L1 and L2 range.

    Args:
        l1_range (float): The L1 range in meters.
        l2_range (float): The L2 range in meters.

    Returns:
        float: The dual channel correction value to be applied to improve accuracy.
    """
    l1_coefficient = 2.546
    l2_coefficient = -1.546

    return l1_coefficient * l1_range + l2_coefficient * l2_range
