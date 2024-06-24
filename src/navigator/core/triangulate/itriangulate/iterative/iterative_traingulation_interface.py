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
import tqdm

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

    def to_standerd_format(
        self,
        solution: dict[str, np.ndarray],
        dops: dict[str, np.ndarray],
        covar: np.ndarray,
    ) -> pd.Series:
        """Convert the solution to a standard format.

        Args:
            solution (dict[str, np.ndarray]): The solution containing the user's position and clock offset.
            dops (dict[str, np.ndarray]): The DOPs.
            covar (np.ndarray): The error covariance matrix.

        Returns:
            pd.Series: The solution in a standard format.
        """
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

        return self.to_standerd_format(solution=solution, dops=dops, covar=covar)

    def wls(
        self,
        epoches: list[Epoch],
        sv_filter: list[str] | None = None,
        use_error_covarience_after: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """The Weighted Least Squares (WLS) method for GPS iterative triangulation.

        This method will estimate the error covariance matrix for the solution using the
        residual error and after a certain number of epochs, it will use inverse of the
        diagonal of the error covariance matrix as the weight matrix for the WLS solver.
        This requires that the all epochs have the same satellites in view as the first epoch.

        Args:
            epoches (list[Epoch]): List of Epoch objects containing observation data and navigation data.
            sv_filter (list[str]): List of satellite PRNs to process.
            estimate_error_covarience (bool): A boolean flag to indicate if the error covariance matrix should be estimated.
            use_error_covarience_after (int): The number of epochs after which the error covariance matrix should be used.
            **kwargs: Additional keyword arguments to pass to the preprocessor.

        Returns:
            pd.DataFrame: The computed iterative triangulation.
        """
        # Initialize the error covariance matrix
        weight = np.eye(len(epoches[0]))
        errorCovarience = np.zeros((len(epoches[0]), len(epoches[0])))

        # Get initial SV filter
        filter = sv_filter if sv_filter is not None else epoches[0].common_sv

        fixes = []

        with tqdm.tqdm(epoches, desc="Iterative Triangulation") as pbar:
            for i, epoch in enumerate(pbar):

                # After a certain number of epochs, use the error covariance matrix
                if i >= use_error_covarience_after:
                    weight = np.linalg.inv(np.diag(np.diag(errorCovarience)))

                # Get the satellite coordinates and the covariance matrix
                solution, covar, dops = self._get_coords_and_covar_matrix(
                    epoch=epoch, weight=weight, sv_filter=filter, **kwargs
                )
                # Calcualte the running average of the error covariance matrix
                errorCovarience = (errorCovarience * i + covar) / (i + 1)

                fixes.append(
                    self.to_standerd_format(solution=solution, dops=dops, covar=covar)
                )

                pbar.set_description(f"Processing {epoch.timestamp}")

        return pd.DataFrame(fixes)
