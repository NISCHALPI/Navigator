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
from .....utils.transforms.coordinate_transforms import geocentric_to_ellipsoidal
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

    def __init__(self, code_only: bool = False) -> None:
        """Initialize the GPSIterativeTriangulationInterface.

        Args:
            code_only (bool): A boolean flag to indicate if the triangulation is code-only i.e no carrier phase measurements are used.

        Returns:
            None
        """
        self.code_only = code_only
        super().__init__(feature="GPS(Iterative)")

    def _get_coords_and_covar_matrix(
        self,
        epoch: Epoch,
        sv_filter: list[str] = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Get the satellite coordinates and the covariance matrix.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            sv_filter (list[str]): List of satellite PRNs to process.
            **kwargs: Additional keyword arguments to pass to the preprocessor.

        Returns:
           tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]: The satellite coordinates, error covariance matrix, and DOPs.
        """
        z, sv_coord = self._preprocess(
            epoch=epoch,
            computational_format=True,
            sv_filter=sv_filter,
            code_only=self.code_only,
            **kwargs,
        )

        # Initial Guess
        x0 = epoch.approximate_coords[["x", "y", "z", "cdt"]].to_numpy(dtype=np.float64)

        # Send to the least squares solver to compute the solution and DOPs
        sols = wls_triangulation(
            pseudorange=z,
            sv_pos=sv_coord,
            W=kwargs.get("weight", np.eye(z.shape[0])).astype(np.float64),
            x0=x0.astype(np.float64),
            max_iter=1000,
            eps=1e-7,
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

        Returns:
            pd.Series | pd.DataFrame: The computed iterative triangulation.
        """
        # Get the satellite coordinates and the covariance matrix
        solution, covar, dops = self._get_coords_and_covar_matrix(epoch=epoch, **kwargs)

        # Calculate Q
        UREA = np.sqrt(np.trace(covar))

        # Convert the geocentric coordinates to ellipsoidal coordinates
        lat, lon, height = geocentric_to_ellipsoidal(
            x=solution[0], y=solution[1], z=solution[2], max_iter=1000
        )

        # Convert the solution
        solution = {
            "x": solution[0],
            "y": solution[1],
            "z": solution[2],
            "cdt": solution[3],
            "lat": lat,
            "lon": lon,
            "height": height,
            "UREA": UREA,
        }

        # Add the DOPs to the solution
        solution.update(dops)

        # Return the solution
        return pd.Series(solution)
