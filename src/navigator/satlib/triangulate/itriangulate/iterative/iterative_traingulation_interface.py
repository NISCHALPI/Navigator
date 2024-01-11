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

from .....utility.epoch import Epoch
from .....utility.transforms.coordinate_transforms import geocentric_to_ellipsoidal
from ..algos.linear_iterative_method import least_squares
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

    def _compute(
        self,
        obs: Epoch,
        obs_metadata: pd.Series,  # noqa: ARG002
        nav_metadata: pd.Series,
        **kwargs,  # noqa: ARG002
    ) -> pd.Series | pd.DataFrame:
        """Compute the iterative triangulation using GPS observations and navigation data.

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.
            obs_metadata (pd.Series): Metadata for the observation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Additional Keyword Arguments:
            approx (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.
            warn (bool, optional): If True, then warning is raised. Defaults to False.

        Returns:
            pd.Series | pd.DataFrame: The computed iterative triangulation.
        """
        # Preprocess the observation and navigation data
        pseduorange, coords = self._preprocess(
            epoch=obs, obs_metadata=obs_metadata, nav_metadata=nav_metadata, **kwargs
        )

        # Send to the least squares solver to compute the solution and DOPs
        solution, covar, sigma = least_squares(
            pseudorange=pseduorange.to_numpy(dtype=np.float64).reshape(-1, 1),
            sv_pos=coords.to_numpy(dtype=np.float64),
            weight=kwargs.get("weight", np.eye(coords.shape[0], dtype=np.float64)),
            eps=1e-6,
        )

        # Calculate Q
        Q = covar / sigma[0, 0] ** 2

        # Calculate the DOPs
        dops = {
            "GDOP": np.sqrt(np.trace(Q)),
            "PDOP": np.sqrt(np.trace(Q[:3, :3])),
            "TDOP": np.sqrt(Q[3, 3]),
            "HDOP": np.sqrt(Q[0, 0] + Q[1, 1]),
            "VDOP": np.sqrt(Q[2, 2]),
            "sigma": sigma[0, 0],
        }

        # Convert the geocentric coordinates to ellipsoidal coordinates
        lat, lon, height = geocentric_to_ellipsoidal(
            x=solution[0, 0], y=solution[1, 0], z=solution[2, 0]
        )

        # Convert the solution
        solution = {
            "x": solution[0, 0],
            "y": solution[1, 0],
            "z": solution[2, 0],
            "dt": solution[3, 0] / 299792458,  # Convert the clock offset to seconds
            "lat": lat,
            "lon": lon,
            "height": height,
        }

        # Add the DOPs to the solution
        solution.update(dops)

        # Return the solution
        return pd.Series(solution)
