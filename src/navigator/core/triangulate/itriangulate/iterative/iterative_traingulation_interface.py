"""The `GPSIterativeTriangulationInterface` module provides classes and methods for GPS iterative triangulation.

It implements functionality for estimating a user's position using GPS observations and navigation data using the least-squares triangulation method.

Author:
    - Nischal Bhattarai (nbhattrai@crimson.ua.edu)

Classes:
    - `GPSIterativeTriangulationInterface`:
        A Interface class for GPS iterative triangulation. Uses Weighted Least Squares to estimate the user's position.

Usage:
    Import this module to access the `GPSIterativeTriangulationInterface` class for GPS-based iterative triangulation.
"""

import numpy as np
import pandas as pd

from .....epoch import Epoch
from .....utility.transforms.coordinate_transforms import geocentric_to_ellipsoidal
from ..algos.wls import wls_triangulation
from ..itriangulate import Itriangulate

__all__ = ["IterativeTriangulationInterface"]


class IterativeTriangulationInterface(Itriangulate):
    """A Interface class for GPS iterative triangulation.

    This class implements the GPS iterative triangulation using GPS observations and navigation data.
    It provides methods for choosing the best navigation message, computing ionospheric corrections,
    emission epochs, satellite coordinates at the emission epoch, and performing least-squares triangulation
    to estimate the user's position.

    Methods:
        __init__:
            Initializes an instance of the GPSIterativeTriangulationInterface class.
        _compute:
            Computes the iterative triangulation using GPS observations and navigation data.

    Attributes:
        None

    """

    def __init__(self) -> None:
        """Initialize the GPSIterativeTriangulationInterface.

        Args:
            None

        Returns:
            None
        """
        super().__init__(feature="GPS(Iterative)")

    def _get_coords_and_covar_matrix(
        self, epoch: Epoch, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Get the satellite coordinates and the covariance matrix.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            **kwargs: Additional keyword arguments.

        Returns:
           tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]: The satellite coordinates, error covariance matrix, and DOPs.
        """
        pseuorange, coords = self._preprocess(epoch=epoch, **kwargs)

        # Initial Guess
        x0 = np.zeros(4)
        if "prior" in kwargs:
            x0 = kwargs["prior"][["x", "y", "z", "dt"]].values
            x0[3] *= 299792458.0

        # Send to the least squares solver to compute the solution and DOPs
        sols = wls_triangulation(
            pseudorange=pseuorange.values,
            sv_pos=coords.values,
            W=kwargs.get("weight", np.eye(pseuorange.shape[0])),
            x0=x0,
            max_iter=1000,
            eps=1e-5,
        )
        return sols["solution"], sols["error_covariance"], sols["dops"]

    def _compute(
        self,
        epoch: Epoch,
        **kwargs,  # noqa: ARG002
    ) -> pd.Series | pd.DataFrame:
        """Compute the iterative triangulation using GPS observations and navigation data.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Additional Keyword Arguments:
            weight (np.ndarray): The weight matrix for the least squares solver.

        Additional Keyword Arguments to GPSPreprocessor:
            mode (str): The mode of operation for the preprocessor. [dual, single]
            prior (pd.Series): The prior estimate for the user's position.
            apply_iono (bool): A flag to apply ionospheric corrections to the pseudorange.
            apply_tropo (bool): A flag to apply tropospheric corrections to the pseudorange.

        Returns:
            pd.Series | pd.DataFrame: The computed iterative triangulation.
        """
        # Get the satellite coordinates and the covariance matrix
        solution, covar, dops = self._get_coords_and_covar_matrix(epoch=epoch, **kwargs)
        # Calculate Q
        UREA = np.sqrt(np.trace(covar))

        # Convert the geocentric coordinates to ellipsoidal coordinates
        lat, lon, height = geocentric_to_ellipsoidal(
            x=solution[0], y=solution[1], z=solution[1], max_iter=1000
        )

        # Convert the solution
        solution = {
            "x": solution[0],
            "y": solution[1],
            "z": solution[2],
            "dt": solution[3],
            "lat": lat,
            "lon": lon,
            "height": height,
            "UREA": UREA,
        }

        # Add the DOPs to the solution
        solution.update(dops)

        # Return the solution
        return pd.Series(solution)
