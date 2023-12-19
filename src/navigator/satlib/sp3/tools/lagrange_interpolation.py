"""This module implements the Lagrange interpolation method."""

import numpy as np

__all__ = ['lagrange_interpolation']


def lagrange_interpolation(
    t: float,
    x_0: np.ndarray,
    y_0: np.ndarray,
) -> float:
    """Function to compute the Lagrange interpolation.

    Uses Lagrange interpolation to create a polynomial interpolation of given points. The method constructs a polynomial of degree n-1 (where n is the number of points) using Lagrange basis polynomials.

    Args:
        t (float): Point at which to interpolate.
        x_0 (np.ndarray): X values.
        y_0 (np.ndarray): Y values.

    Raises:
        ValueError: If x_0 and y_0 are not equal and 1D arrays.

    Returns:
        float: Value at t.
    """
    # Assert equal and 1D arrays
    if len(x_0) != len(y_0) and x_0.ndim != 1 and y_0.ndim != 1:
        raise ValueError('x_0 and y_0 must be equal and 1D arrays.')

    # Initialize value
    value = 0.0

    # Loop over all points
    for i in range(len(y_0)):
        numerator = t - np.delete(x_0, i)
        denominator = x_0[i] - np.delete(x_0, i)

        # Add to value
        value += y_0[i] * np.prod(numerator) / np.prod(denominator)

    return value
