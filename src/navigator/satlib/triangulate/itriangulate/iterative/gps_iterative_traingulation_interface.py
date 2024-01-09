"""The `GPSIterativeTriangulationInterface` module provides classes and methods for GPS iterative triangulation.

It implements functionality for estimating a user's position using GPS observations and navigation data using the least-squares triangulation method.

Author:
    - Nischal Bhattarai (nbhattrai@crimson.ua.edu)

Classes:
    - `GPSIterativeTriangulationInterface`:
        Implements iterative triangulation using GPS observations and navigation data.
        Methods include computing ionospheric corrections, emission epochs,
        satellite coordinates, and performing least-squares triangulation to estimate user position.

Usage:
    Import this module to access the `GPSIterativeTriangulationInterface` class for GPS-based iterative triangulation.
"""

from warnings import warn

import numpy as np
import pandas as pd

from .....utility.epoch import Epoch
from .....utility.transforms.coordinate_transforms import geocentric_to_ellipsoidal
from ....satellite.iephm.sv.igps_ephm import IGPSEphemeris
from ....satellite.satellite import Satellite
from ..algos.dual_frequency_corrections import dual_channel_correction
from ..algos.linear_iterative_method import least_squares
from ..algos.rotations import earth_rotation_correction
from ..itriangulate import Itriangulate

__all__ = ["GPSIterativeTriangulationInterface"]


class GPSIterativeTriangulationInterface(Itriangulate):
    """A Interface class for GPS iterative triangulation.

    This class implements the GPS iterative triangulation using GPS observations and navigation data.
    It provides methods for choosing the best navigation message, computing ionospheric corrections,
    emission epochs, satellite coordinates at the emission epoch, and performing least-squares triangulation
    to estimate the user's position.

    Methods:
        __init__:
            Initialize the GPSIterativeTriangulationInterface.
        _ionospehric_free_combination:
            Compute the Ionospheric free combination for GPS observations.
        _compute_sv_coordinates_at_emission_epoch:
            Computes the satellite coordinates at the emission epoch.
        _rotate_satellite_coordinates_to_reception_epoch:
            Rotate the satellite coordinates to the reception epoch.
        _compute:
            Compute the iterative triangulation using GPS observations and navigation data.

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

    def _ionospehric_free_combination(
        self, obs_data: pd.DataFrame, code_warnings: bool = True
    ) -> pd.Series:
        """Compute the Ionospheric free combination for GPS observations.

        Args:
            obs_data (pd.DataFrame): GPS observations data.
            code_warnings (bool, optional): If True, then warning is raised. Defaults to True.

        Returns:
            pd.DataFrame: The computed ionospheric free combination pseudorange for GPS observations.
        """
        pseudorange = None  # Initialize the pseudorange to None

        # If C1C and C2C are both present, then compute the ionospheric correction wrt C1C and C2C
        if "C1C" in obs_data.columns and "C2C" in obs_data.columns:
            pseudorange = dual_channel_correction(obs_data["C1C"], obs_data["C2C"])
        # If C1W and C2W are both present, then compute the ionospheric correction wrt C1W and C2W
        elif "C1W" in obs_data.columns and "C2W" in obs_data.columns:
            if code_warnings:
                warn(
                    message="C1W and C2W are used for ionospheric correction. This is not recommended."  # Check if this is correct
                )
            pseudorange = dual_channel_correction(obs_data["C1W"], obs_data["C2W"])
        elif "C1C" in obs_data.columns and "C2W" in obs_data.columns:
            if code_warnings:
                warn(
                    message="C1C and C2W are used for ionospheric correction. This is not recommended."  # Check if this is correct
                )
            pseudorange = dual_channel_correction(obs_data["C1C"], obs_data["C2W"])

        else:
            raise ValueError(
                "Invalid observation data. Dual Frequency Ion Free Combination not applied."
            )

        return pseudorange

    def _compute_sv_coordinates_at_emission_epoch(
        self,
        reception_time: pd.Timestamp,
        pseudorange: pd.Series,
        nav: pd.DataFrame,
        nav_metadata: pd.Series,
        **kwargs,
    ) -> Epoch:
        """Computes the satellite coordinates at the emission epoch.

        This method computes the satellite coordinates at the emission epoch using the GPS observations and navigation data.
        It instantiates the Satellite class and computes the satellite coordinate at the emission epoch.

        Args:
            reception_time (pd.Timestamp): Reception time of the GPS observations.
            pseudorange (pd.Series): Pseudorange of the GPS observations.
            nav (pd.DataFrame): Navigation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            **kwargs: Additional keyword arguments to be passed to the Satellite class.

        Returns:
            Epoch: The computed satellite coordinates at the emission epoch.
        """
        # Compute the emission epoch
        dt = pseudorange / 299792458

        # Compute the emission epoch
        emission_epoch = reception_time - pd.to_timedelta(dt, unit="s")

        # Instantiate the Satellite class
        satellite = Satellite(iephemeris=IGPSEphemeris())

        # t_sv must have same indexed dataframes as nav. Compatibility check!!
        t_sv = pd.DataFrame(
            index=nav.index, columns=["Tsv"], data=emission_epoch.to_numpy()
        )

        # Compute the satellite coordinate at the emission epoch
        return satellite(
            t_sv=t_sv, metadata=nav_metadata, data=nav, **kwargs
        ).droplevel("time")

    def _rotate_satellite_coordinates_to_reception_epoch(
        self,
        sv_coords: pd.DataFrame,
        pseudorange: pd.Series,
        approx_receiver_location: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Rotate the satellite coordinates to the reception epoch.

        This method rotates the satellite coordinates to the reception epoch using the Earth rotation correction.

        Methods:
            - Rotate by following angule for each satellite using the omega_e * (pseudorange / speed of light).

        Args:
            sv_coords (pd.DataFrame): Satellite coordinates at the emission epoch.
            pseudorange (pd.Series): Pseudorange of the GPS observations.
            approx_receiver_location (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.

        Returns:
            Epoch: The rotated satellite coordinates at the reception epoch.
        """
        dt = None  # Initialize the dt to None
        if approx_receiver_location is None:
            # Compute the dt for each satellite naively
            # Use the pseudorange and the speed of light to compute the dt
            # Include the satellite clock correction and other unmodeled effects
            dt = pseudorange / 299792458
        else:
            # Check if ['x', 'y', 'z'] are present in the approx_receiver_location
            if not all(
                [coord in approx_receiver_location.index for coord in ['x', 'y', 'z']]
            ):
                raise ValueError(
                    "Invalid approx_receiver_location. Must contain ['x', 'y', 'z'] coordinates."
                )
            # Compute the dt using method in Equation 5.13 in ESA GNSS Book
            # https://gssc.esa.int/navipedia/GNSS_Book/ESA_GNSS-Book_TM-23_Vol_I.pdf
            dt = (
                sv_coords[['x', 'y', 'z']] - approx_receiver_location[['x', 'y', 'z']]
            ).apply(lambda row: row.dot(row) ** 0.5 / 299792458, axis=1)

        # Rotate the satellite coordinates to the reception epoch
        sv_coords[['x', 'y', 'z']] = earth_rotation_correction(
            sv_position=sv_coords[['x', 'y', 'z']].to_numpy(dtype=np.float64),
            dt=dt.to_numpy(dtype=np.float64).ravel(),
        )

        return sv_coords

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
        # Use Epoch to get the navigation message for the observation epoch. Held at "Epoch.nav_data" attribute
        obs_data, nav_data = obs.obs_data, obs.nav_data

        # Compute the ionospheric free combination
        # Populates the 'Pseudorange' column in obs.obs_data
        pseduorange = self._ionospehric_free_combination(
            obs_data=obs_data, code_warnings=kwargs.get("warn", False)
        )

        # Compute the satellite coordinates at the emission epoch
        # This also computes satellite clock correction which is stored in the 'dt' column.
        coords = self._compute_sv_coordinates_at_emission_epoch(
            reception_time=obs.timestamp,
            pseudorange=pseduorange,
            nav=nav_data,
            nav_metadata=nav_metadata,
            **kwargs,
        )

        # Need to apply the earth rotation correction since SV coordinates are in ECEF in emission epoch
        # Need to rotate each satellite coordinate to the reception epoch since it is common epoch for all satellites
        coords = self._rotate_satellite_coordinates_to_reception_epoch(
            sv_coords=coords,
            pseudorange=pseduorange,
            approx_receiver_location=kwargs.get("approx", None),
        )

        # Correct the pseudorange for the satellite clock offset.
        # This crossponds to the satellite clock correction. P(j) + c * dt(j)
        pseduorange += coords["dt"] * 299792458

        # Send to the least squares solver to compute the solution and DOPs
        solution, covar, sigma = least_squares(
            pseudorange=pseduorange.to_numpy(dtype=np.float64).reshape(-1, 1),
            sv_pos=coords[['x', 'y', 'z']].to_numpy(dtype=np.float64),
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
            "dt": solution[3, 0],
            "lat": lat,
            "lon": lon,
            "height": height,
        }

        # Add the DOPs to the solution
        solution.update(dops)

        # Return the solution
        return pd.Series(solution)
