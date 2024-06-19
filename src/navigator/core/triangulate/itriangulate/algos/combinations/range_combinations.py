"""GNSS Pseudo-Range Combinations Module.

This module provides functions for calculating various pseudo-range combinations used in GNSS (Global Navigation Satellite System) triangulation algorithms. Pseudo-ranges are measurements of the signal travel time from satellites to a receiver, and their combinations help mitigate certain error sources.

For more information on the combination of GNSS measurements, refer to the official documentation:
https://gssc.esa.int/navipedia/index.php/Combination_of_GNSS_Measurements

Functions:
- ionosphere_free_combination(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    Computes the ionosphere-free combination of pseudo-ranges.

- geometry_free_combination(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    Computes the geometry-free combination of pseudo-ranges.

- wide_lane_combination(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    Computes the wide-lane combination of pseudo-ranges.

- narrow_lane_combination(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    Computes the narrow-lane combination of pseudo-ranges.

Constants:
- L1_FREQ: Frequency of the L1 GNSS signal (1575.42 MHz).
- L2_FREQ: Frequency of the L2 GNSS signal (1227.60 MHz).

Usage:
Import the module and use the provided functions to calculate different pseudo-range combinations.

Example:
```python
import navigator.core.triangulate.itriangulate.algos.combinations.range_combinations as gpr

p1 = np.array([100.0, 150.0, 200.0])
p2 = np.array([120.0, 130.0, 180.0])

iono_free_result = gpr.ionosphere_free_combination(p1, p2)
geo_free_result = gpr.geometry_free_combination(p1, p2)
wide_lane_result = gpr.wide_lane_combination(p1, p2)
narrow_lane_result = gpr.narrow_lane_combination(p1, p2)
"""

import numpy as np

# Constants
L1_FREQ = 1575.42e6
L2_FREQ = 1227.60e6
SPEED_OF_LIGHT = 299792458.0
L1_WAVELENGTH = 0.19029367279836487
L2_WAVELENGTH = 0.24421021342456825
IF_WAVELENGTH = 0.10695337814214669


# ION FREE WAVELNGTH
ION_FREE_WAVELENGTH = 2


__all__ = [
    "ionosphere_free_combination",
    "geometry_free_combination",
    "wide_lane_combination",
    "narrow_lane_combination",
]


def ionosphere_free_combination(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    r"""Computes the ionosphere-free combination of pseudo-ranges.

    Args:
        p1 (np.ndarray): Pseudo-range measurements for the first frequency.
        p2 (np.ndarray): Pseudo-range measurements for the second frequency.

    Returns:
        np.ndarray: Ionosphere-free combination of pseudo-ranges.

    """
    return (L1_FREQ**2 * p1 - L2_FREQ**2 * p2) / (L1_FREQ**2 - L2_FREQ**2)


def geometry_free_combination(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    r"""Computes the geometry-free combination of pseudo-ranges.

    Args:
        p1 (np.ndarray): Pseudo-range measurements for the first frequency.
        p2 (np.ndarray): Pseudo-range measurements for the second frequency.

    Returns:
        np.ndarray: Geometry-free combination of pseudo-ranges.

    """
    return p1 - p2


def wide_lane_combination(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    r"""Computes the wide-lane combination of pseudo-ranges.

    Args:
        p1 (np.ndarray): Pseudo-range measurements for the first frequency.
        p2 (np.ndarray): Pseudo-range measurements for the second frequency.

    Returns:
        np.ndarray: Wide-lane combination of pseudo-ranges.

    """
    return (L1_FREQ * p1 - L2_FREQ * p2) / (L1_FREQ - L2_FREQ)


def narrow_lane_combination(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    r"""Computes the narrow-lane combination of pseudo-ranges.

    Args:
        p1 (np.ndarray): Pseudo-range measurements for the first frequency.
        p2 (np.ndarray): Pseudo-range measurements for the second frequency.

    Returns:
        np.ndarray: Narrow-lane combination of pseudo-ranges.

    """
    return (L1_FREQ * p1 + L2_FREQ * p2) / (L1_FREQ + L2_FREQ)
