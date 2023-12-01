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

import pandas as pd

from .....utility.epoch import Epoch
from .....utility.transforms.coordinate_transforms import geocentric_to_ellipsoidal
from ....iephm.sv.igps_ephm import IGPSEphemeris
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

    def _ionospehric_free_combination(self, obs: Epoch, no_warn: bool = True) -> Epoch:
        """Compute the Ionospheric free combination for GPS observations.

        Args:
            obs (Epoch): GPS observations.
            no_warn (bool, optional): If True, then no warning is raised. Defaults to True.

        Returns:
            Epoch: GPS observations with ionospheric correction.
        """
        corrected_obs = obs.obs_data.copy()
        # If C1C and C2C are both present, then compute the ionospheric correction wrt C1C and C2C
        if "C1C" in obs.obs_data.columns and "C2C" in obs.obs_data.columns:
            corrected_obs["Pseudorange"] = dual_channel_correction(
                obs.obs_data["C1C"], obs.obs_data["C2C"]
            )
        # If C1W and C2W are both present, then compute the ionospheric correction wrt C1W and C2W
        elif "C1W" in obs.obs_data.columns and "C2W" in obs.obs_data.columns:
            if not no_warn:
                warn(
                    message="C1W and C2W are used for ionospheric correction. This is not recommended."  # Check if this is correct
                )
            corrected_obs["Pseudorange"] = dual_channel_correction(
                obs.obs_data["C1W"], obs.obs_data["C2W"]
            )
        elif "C1C" in obs.obs_data.columns and "C2W" in obs.obs_data.columns:
            if not no_warn:
                warn(
                    message="C1C and C2W are used for ionospheric correction. This is not recommended."  # Check if this is correct
                )
            corrected_obs["Pseudorange"] = dual_channel_correction(
                obs.obs_data["C1C"], obs.obs_data["C2W"]
            )

        else:
            raise ValueError(
                "Invalid observation data. Dual Frequency Ion Free Combination not applied."
            )

        # Replace the Pseudorange column in obs.data with the corrected Pseudorange
        obs.obs_data = corrected_obs

        return obs

    def _compute_sv_coordinates_at_emission_epoch(
        self,
        obs: Epoch,
        nav: pd.DataFrame,
        nav_metadata: pd.Series,
    ) -> Epoch:
        """Computes the satellite coordinates at the emission epoch.

        This method computes the satellite coordinates at the emission epoch using the GPS observations and navigation data.
        It instantiates the Satellite class and computes the satellite coordinate at the emission epoch.

        Args:
            obs (Epoch): GPS observations.
            obs_metadata (pd.Series): Metadata for the GPS observations.
            nav (pd.DataFrame): Navigation data.
            nav_metadata (pd.Series): Metadata for the navigation data.

        Returns:
            Epoch: The computed satellite coordinates at the emission epoch.
        """
        # Compute the emission epoch
        dt = obs.obs_data["Pseudorange"] / 299792458

        # Compute the emission epoch
        emission_epoch = obs.timestamp - pd.to_timedelta(dt, unit="s")
        emission_epoch.name = "EmissionEpoch"

        # Instantiate the Satellite class
        satellite = Satellite(iephemeris=IGPSEphemeris())

        # t_sv must have same indexed dataframes as nav. Compatibility check!!
        t_sv = (
            emission_epoch.to_frame()
            .join(nav)[["EmissionEpoch"]]
            .rename({"EmissionEpoch": "Tsv"}, axis=1)
        )

        # Compute the satellite coordinate at the emission epoch
        return satellite(t_sv=t_sv, metadata=nav_metadata, data=nav).droplevel("time")

    def _rotate_satellite_coordinates_to_reception_epoch(
        self,
        sv_coords: pd.DataFrame,
        obs_data: pd.DataFrame,
        approx_coords: pd.Series = None,
    ) -> pd.DataFrame:
        """Rotate the satellite coordinates to the reception epoch.

        This method rotates the satellite coordinates to the reception epoch using the Earth rotation correction.

        Methods:
            - Rotate by following angule for each satellite using the omega_e * (pseudorange / speed of light). (Not Preffered)
            - If the user provides approximate coordinates of receiver, then rotation is done by omega_e * (|approx - satellite_coord| / speed of light).

        Args:
            sv_coords (pd.DataFrame): Satellite coordinates at the emission epoch.
            obs_data (pd.DataFrame): Observation data containing the pseudorange.
            approx_coords (pd.Series, optional): Approximate coordinates of the receiver. Defaults to None.

        Returns:
            Epoch: The rotated satellite coordinates at the reception epoch.
        """
        # Rotation by dt time for each satellite
        dt = 0

        if approx_coords is not None:
            if not all([coord in approx_coords.index for coord in ["x", "y", "z"]]):
                raise ValueError(
                    "Approximate coordinates must contain x, y, and z coordinates."
                )

            # Compute the dt for each satellite
            dt = (
                (sv_coords[['x', 'y', 'z']] - approx_coords[['x', 'y', 'z']]) ** 2
            ).sum(axis=1) ** 0.5 / 299792458

        else:
            # Compute the dt for each satellite naively
            dt = obs_data['Pseudorange'] / 299792458

        # Rotate the satellite coordinates to the reception epoch
        sv_coords[['x', 'y', 'z']] = earth_rotation_correction(
            sv_position=sv_coords[['x', 'y', 'z']].to_numpy(), dt=dt.to_numpy().ravel()
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


        Returns:
            pd.Series | pd.DataFrame: The computed iterative triangulation.
        """
        # Use Epoch to get the navigation message for the observation epoch. Held at "Epoch.nav_data" attribute
        obs, nav = obs, obs.nav_data

        # Compute the ionospheric free combination
        # Populates the 'Pseudorange' column in obs.obs_data
        obs = self._ionospehric_free_combination(obs)

        # Compute the satellite coordinates at the emission epoch
        # This also computes satellite clock correction which is stored in the 'dt' column.
        coords = self._compute_sv_coordinates_at_emission_epoch(obs, nav, nav_metadata)

        # Need to apply the earth rotation correction since SV coordinates are in ECEF in emission epoch
        # Need to rotate each satellite coordinate to the reception epoch since it is common epoch for all satellites
        # This is delegated to a separate method because there are two ways to rotate which depends on if the reciever location is known or not!
        # See self._rotate_satellite_coordinates_to_reception_epoch for more details.
        coords = self._rotate_satellite_coordinates_to_reception_epoch(
            sv_coords=coords,
            obs_data=obs.obs_data,
            approx_coords=pd.Series(kwargs["approx_position"])
            if 'approx_position' in kwargs
            else None,
        )

        # Attach the relevant statistics to a new frame that contains
        # pseudorange and sv coordinates
        stats = obs.obs_data[["Pseudorange"]].join(coords)

        # Correct the pseudorange for the satellite clock offset.
        # This crossponds to the satellite clock correction. P(j) + c * dt(j)
        stats["Pseudorange"] = stats["Pseudorange"] + stats["dt"] * 299792458

        # If less than 4 satellites are available, then raise an error
        if stats.shape[0] < 4:
            raise ValueError(
                f"Insufficient number of satellites. Only {stats.shape[0]} satellites observed in the epoch."
            )

        # Extract the
        Range, Coords = (
            stats["Pseudorange"].to_numpy().reshape(-1, 1),
            stats[["x", "y", "z"]].to_numpy(),
        )

        # Send to the least squares solver to compute the solution and DOPs
        dops, solution = least_squares(Range, Coords, eps=1e-6)

        # Convert the geocentric coordinates to ellipsoidal coordinates
        lat, lon, height = geocentric_to_ellipsoidal(*solution[:3])

        # Extract the solution into a series
        return pd.Series(
            {
                "x": solution[0],
                "y": solution[1],
                "z": solution[2],
                "dt": solution[3],
                "lat": lat,
                "lon": lon,
                "height": height,
                "GDOP": dops["GDOP"],
                "PDOP": dops["PDOP"],
                "TDOP": dops["TDOP"],
            }
        )
