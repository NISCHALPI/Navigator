"""Triangulate using GPS observations and navigation data.

This module provides a class, GPSIterativeTriangulationInterface, which implements iterative triangulation using GPS observations and navigation data. It offers methods to choose the best navigation message, compute ionospheric corrections, emission epochs, and satellite coordinates at the emission epoch, and perform least-squares triangulation to estimate the user's position.

Classes:
    - GPSIterativeTriangulationInterface: A class for GPS iterative triangulation.

Functions:
    - None

Attributes:
    - __all__: A list of names that are exported when using 'from module import *'.

Dependencies:
    - pandas: Used for handling data in DataFrame format.
    - epoch: Imported from a utility module to handle timestamped data.
    - iephm.igps_ephm: Importing IGPSEphemeris class for satellite ephemeris data.
    - satellite.satellite: Importing Satellite class for satellite position calculations.
    - algos.dual_frequency_corrections: Provides dual-channel ionospheric corrections.
    - algos.linear_iterative_method: Provides the least-squares solver for triangulation.
    - algos.rotations: Includes functions for Earth rotation corrections.
    - itriangulate: An interface for triangulation.

Public Classes:
    - GPSIterativeTriangulationInterface: A class for GPS iterative triangulation that implements Itriangulate.

Public Methods (GPSIterativeTriangulationInterface):
    - __init__(self) -> None: Initializes the GPSIterativeTriangulationInterface.

    - _choose_nav_message_for_interpolation(self, obs: Epoch, nav: pd.DataFrame, ephemeris: str = "maxsv") -> pd.DataFrame:
        Choose the best navigation message based on method. Method can be "nearest" or "maxsv," which chooses the "nearest" nav timestamp or timestamp containing the maximum number of satellites.

    - _ionospheric_correction(self, obs: Epoch) -> Epoch:
        Compute the Ionospheric correction for the GPS observations.

    - _compute_emission_epoch(self, obs: Epoch) -> Epoch:
        Compute the emission epoch for the GPS observations.

    - _compute_sv_coordinates_at_emission_epoch(self, obs: Epoch, nav: pd.DataFrame, nav_metadata: pd.Series) -> Epoch:
        Compute satellite coordinates at the emission epoch.

    - _compute(self, obs: Epoch, obs_metadata: pd.Series, nav: pd.DataFrame, nav_metadata: pd.Series, **kwargs) -> pd.Series | pd.DataFrame:
        Compute the iterative triangulation using GPS observations and navigation data.
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

    Args:
        Itriangulate (type): The base class for triangulation.

    Public Methods:
        - __init__(self) -> None:
            Initializes the GPSIterativeTriangulationInterface.

        - _choose_nav_message_for_interpolation(self, obs: Epoch, nav: pd.DataFrame, ephemeris: str = "maxsv") -> pd.DataFrame:
            Choose the best navigation message based on the specified method.

        - _ionospheric_correction(self, obs: Epoch) -> Epoch:
            Compute the Ionospheric correction for GPS observations.

        - _compute_emission_epoch(self, obs: Epoch) -> Epoch:
            Compute the emission epoch for GPS observations.

        - _compute_sv_coordinates_at_emission_epoch(self, obs: Epoch, nav: pd.DataFrame, nav_metadata: pd.Series) -> Epoch:
            Compute satellite coordinates at the emission epoch.

        - _compute(self, obs: Epoch, obs_metadata: pd.Series, nav: pd.DataFrame, nav_metadata: pd.Series, **kwargs) -> pd.Series | pd.DataFrame:
            Compute the iterative triangulation using GPS observations and navigation data.
    """

    def __init__(self) -> None:
        """Initialize the GPSIterativeTriangulationInterface.

        Args:
            None

        Returns:
            None
        """
        super().__init__(feature="GPS(Iterative)")

    def _choose_nav_message_for_interpolation(
        self,
        obs: Epoch,
        nav: pd.DataFrame,
        ephemeris: str = "maxsv",
    ) -> tuple[Epoch, pd.DataFrame]:
        """Choose the best navigation message based on the specified method.

        Args:
            obs (Epoch): GPS observations.
            nav (pd.DataFrame): Navigation data.
            ephemeris (str, optional): Method to choose the best navigation message. Defaults to 'maxsv'.

        Returns:
            tuple[Epoch,  pd.DataFrame]: The chosen navigation message and the observations with the intersected ephemeris.
        """
        # Get the epoch time
        epoch_time = obs.timestamp
        navtimes = nav.index.get_level_values("time").unique()

        # If the max time difference is greater that 4hours between epoch time and nav time,
        # then raise an error since ephemeris is not valid for that time
        if all(abs(navtimes - epoch_time) > pd.Timedelta("4h")):
            raise ValueError(
                f"No valid ephemeris for {epoch_time}. All ephemeris are more than 4 hours away from the epoch time."
            )

        if ephemeris.lower() == "nearest":
            # Get the nearest timestamp from epoch_time
            nearest_time = min(navtimes, key=lambda x: abs(x - epoch_time))
            ephm = nav.loc[[nearest_time]]

        elif ephemeris.lower() == "maxsv":
            # Get the timestamp with maximum number of sv
            maxsv_time = max(navtimes, key=lambda x: nav.loc[x].shape[0])
            ephm = nav.loc[[maxsv_time]]

        else:
            raise ValueError(
                'Invalid ephemeris method. Method must be "nearest" or "maxsv".'
            )

        # Intersect the ephemeris with the observations
        intersection = obs.data.index.intersection(ephm.index.get_level_values("sv"))
        ephm = ephm.loc[(slice(None), intersection), :]

        # Create a new obs epoch with the intersected ephemeris
        intersected_obs = Epoch(
            timestamp=obs.timestamp,
            data=obs.data.loc[intersection, :],
        )

        # If the intersection is empty, then raise an error
        if ephm.empty:
            raise ValueError(
                "Use maxsv ephemeris mode. No common sv between observations and nav ephemeris data."
            )

        return intersected_obs, ephm

    def _ionospehric_correction(self, obs: Epoch) -> Epoch:
        """Compute the Ionospheric correction for GPS observations.

        Args:
            obs (Epoch): GPS observations.

        Returns:
            Epoch: GPS observations with ionospheric correction.
        """
        corrected_obs = obs.data.copy()
        # If C1C and C2C are both present, then compute the ionospheric correction wrt C1C and C2C
        if "C1C" in obs.data.columns and "C2C" in obs.data.columns:
            corrected_obs["Pseudorange"] = dual_channel_correction(
                obs.data["C1C"], obs.data["C2C"]
            )
        # If C1W and C2W are both present, then compute the ionospheric correction wrt C1W and C2W
        elif "C1W" in obs.data.columns and "C2W" in obs.data.columns:
            warn(
                message="C1W and C2W are used for ionospheric correction. This is not recommended."  # Check if this is correct
            )
            corrected_obs["Pseudorange"] = dual_channel_correction(
                obs.data["C1W"], obs.data["C2W"]
            )
        elif "C1C" in obs.data.columns and "C2W" in obs.data.columns:
            warn(
                message="C1C and C2W are used for ionospheric correction. This is not recommended."  # Check if this is correct
            )
            corrected_obs["Pseudorange"] = dual_channel_correction(
                obs.data["C1C"], obs.data["C2W"]
            )

        else:
            raise ValueError(
                "Invalid observation data. Dual Frequency Ion Free Combination not applied."
            )

        # Replace the Pseudorange column in obs.data with the corrected Pseudorange
        obs._data = corrected_obs

        return obs

    def _compute_emission_epoch(self, obs: Epoch) -> Epoch:
        """Compute satellite coordinates at the emission epoch.

        Args:
            obs (Epoch): GPS observations.
            nav (pd.DataFrame): Navigation data.
            nav_metadata (pd.Series): Metadata for navigation data.

        Returns:
            Epoch: Computed satellite coordinates at the emission epoch.
        """
        # Compute the emission epoch
        obs.data["dt"] = obs.data["Pseudorange"] / 299792458

        # Compute the emission epoch
        obs.data["EmissionEpoch"] = obs.timestamp - pd.to_timedelta(
            obs.data["dt"], unit="s"
        )

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
        # Instantiate the Satellite class
        satellite = Satellite(iephemeris=IGPSEphemeris())

        # t_sv must have same indexed dataframes as nav. Compatibility check!!
        t_sv = (
            obs.data["EmissionEpoch"]
            .to_frame()
            .join(nav)[["EmissionEpoch"]]
            .rename({"EmissionEpoch": "Tsv"}, axis=1)
        )

        # Compute the satellite coordinate at the emission epoch
        return satellite(t_sv=t_sv, metadata=nav_metadata, data=nav).droplevel("time")

    def _compute(
        self,
        obs: Epoch,
        obs_metadata: pd.Series,  # noqa: ARG002
        nav: pd.DataFrame,
        nav_metadata: pd.Series,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Compute the iterative triangulation using GPS observations and navigation data.

        Args:
            obs (Epoch): GPS observations epoch.
            obs_metadata (pd.Series): Metadata for GPS observations.
            nav (pd.DataFrame): Navigation data.
            nav_metadata (pd.Series): Metadata for navigation data.
            **kwargs: Arbitrary keyword arguments (ephemeris = "nearest" | "maxsv").

        Returns:
            pd.Series | pd.DataFrame: The computed iterative triangulation.
        """
        # Get timestamps from the navigation file that are common with the observations
        obs, nav = self._choose_nav_message_for_interpolation(
            obs,
            nav,
            ephemeris="maxsv" if "ephemeris" not in kwargs else kwargs["ephemeris"],
        )

        # Compute the ionospheric free combination
        obs = self._ionospehric_correction(obs)
        # Compute the emission epoch
        obs = self._compute_emission_epoch(obs)

        # Compute the satellite coordinates at the emission epoch
        coords = self._compute_sv_coordinates_at_emission_epoch(obs, nav, nav_metadata)

        # Compute the earth rotation correction since SV coordinates are in ECEF
        # Since the coordinates are computed at the emission epoch, the delta_t is the time
        # difference between the emission epoch and the current time
        coords[["x", "y", "z"]] = earth_rotation_correction(
            coords[["x", "y", "z"]].to_numpy(), obs.data["dt"].to_numpy().ravel()
        )

        # Attach the relevant statistics to a new frame that contains
        # pseudorange and sv coordinates
        stats = obs.data[["Pseudorange"]].join(coords)

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
