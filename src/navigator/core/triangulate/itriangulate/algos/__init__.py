"""Import algorithms for triangulation.

This module contains functions to calculate corrections for the triangulation algorithm. The corrections include ionospheric delay, tropospheric delay, and dual-frequency corrections.

Functions:
    - dual_channel_correction: Calculate the dual-frequency correction for the triangulation algorithm.
    - klobuchar_ionospheric_correction: Calculate the ionospheric delay in the signal.
    - least_squares: Calculate the least squares solution for the triangulation algorithm.
    - tropospheric_delay_correction: Calculate the tropospheric delay in the signal.

Note:
    This is a functional interface for the triangulation algorithm. It provides tools for calculating corrections to the signal, including ionospheric delay, tropospheric delay, and dual-frequency corrections

"""

from .ionosphere.dual_frequency_corrections import dual_channel_correction
from .ionosphere.klobuchar_ionospheric_model import klobuchar_ionospheric_correction
from .linear_iterative_method import least_squares
from .troposphere.tropospheric_delay import tropospheric_delay_correction