"""Triangulate using GPS observations and navigation data."""

import pandas as pd

from .....utility.epoch import Epoch
from ....iephm.sv.igps_ephm import IGPSEphemeris
from ....satellite.satellite import Satellite
from ..algos.dual_frequency_corrections import dual_channel_correction
from ..algos.rotations import earth_rotation_correction
from ..itriangulate import Itriangulate

__all__ = ["GPSIterativeTriangulationInterface"]


class GPSIterativeTriangulationInterface(Itriangulate):
    """_summary_.

    Args:
        Itriangulate (_type_): _description_
    """

    def __init__(self) -> None:
        """_summary_."""
        super().__init__(feature="GPS(Iterative)")

    def _choose_nav_message_for_interpolation(
        self,
        obs: Epoch,
        nav: pd.DataFrame,
        ephemeris: str = 'maxsv',
    ) -> pd.DataFrame:
        """Choose the best navigation message based on method. Method can be "nearest"  or "maxsv" which chooses the "nearest" nav timestamp or timestamp containing maximum sv.

        Args:
            obs (Epoch): GPS observations.
            nav (pd.DataFrame): Navigation data.
            ephemeris (str, optional): Method to choose the best navigation message. Defaults to 'maxsv'.

        Returns:
            pd.DataFrame: Dataframe of the gps ephemeris.
        """
        # Get the epoch time
        epoch_time = obs.timestamp
        navtimes = nav.index.get_level_values('time').unique()

        # If the max time difference is greater that 4hours between epoch time and nav time,
        # then raise an error since ephemeris is not valid for that time
        if all(abs(navtimes - epoch_time) > pd.Timedelta('4h')):
            raise ValueError(
                f'No valid ephemeris for {epoch_time}. All ephemeris are more than 4 hours away from the epoch time.'
            )

        if ephemeris.lower() == 'nearest':
            # Get the nearest timestamp from epoch_time
            nearest_time = min(navtimes, key=lambda x: abs(x - epoch_time))
            ephm = nav.loc[[nearest_time]]

        elif ephemeris.lower() == 'maxsv':
            # Get the timestamp with maximum number of sv
            maxsv_time = max(navtimes, key=lambda x: nav.loc[x].shape[0])
            ephm = nav.loc[[maxsv_time]]

        else:
            raise ValueError(
                'Invalid ephemeris method. Method must be "nearest" or "maxsv".'
            )

        # Intersect the ephemeris with the observations
        intersection = obs.data.index.intersection(ephm.index.get_level_values('sv'))
        ephm = ephm.loc[(slice(None), intersection), :]

        # If the intersection is empty, then raise an error
        if ephm.empty:
            raise ValueError(
                'Use maxsv ephemeris mode. No common sv between observations and nav ephemeris data.'
            )

        return ephm

    def _ionospehric_correction(self, obs: Epoch) -> Epoch:
        """Computes the Ionospheric correction for the GPS observations.

        Args:
            obs (Epoch): GPS observations.

        Returns:
            Epoch: GPS observations with ionospheric correction.
        """
        corrected_obs = obs.data.copy()
        # If C1C and C2C are both present, then compute the ionospheric correction wrt C1C and C2C
        if 'C1C' in obs.data.columns and 'C2C' in obs.data.columns:
            corrected_obs['Pseudorange'] = dual_channel_correction(
                obs.data['C1C'], obs.data['C2C']
            )
        # If C1W and C2W are both present, then compute the ionospheric correction wrt C1W and C2W
        elif 'C1W' in obs.data.columns and 'C2W' in obs.data.columns:
            corrected_obs['Pseudorange'] = dual_channel_correction(
                obs.data['C1W'], obs.data['C2W']
            )

        else:
            raise ValueError(
                'Invalid observation data. Dual Frequency Ion Free Combination not applied.'
            )

        # Replace the Pseudorange column in obs.data with the corrected Pseudorange
        obs._data = corrected_obs

        return obs

    def _compute_emission_epoch(self, obs: Epoch) -> Epoch:
        """Computes the emission epoch for the GPS observations.

        Args:
            obs (Epoch): GPS observations.

        Returns:
            pd.Series: The computed emission epoch.
        """
        # Compute the emission epoch
        obs.data['dt'] = obs.data['Pseudorange'] / 299792458

        # Compute the emission epoch
        obs.data['EmissionEpoch'] = obs.timestamp - pd.to_timedelta(
            obs.data['dt'], unit='s'
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
            obs.data['EmissionEpoch']
            .to_frame()
            .join(nav)[['EmissionEpoch']]
            .rename({'EmissionEpoch': 'Tsv'}, axis=1)
        )

        # Compute the satellite coordinate at the emission epoch
        return satellite(t_sv=t_sv, metadata=nav_metadata, data=nav).droplevel('time')

    def _compute(
        self,
        obs: Epoch,
        obs_metadata: pd.Series,
        nav: pd.DataFrame,
        nav_metadata: pd.Series,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Computes the iterative triangulation using GPS observations and navigation data.

        Args:
            obs (Epoch): GPS observations epoch.
            obs_metadata (pd.Series): Metadata for the GPS observations.
            nav (pd.DataFrame): Navigation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments (ephemeris = "nearest" | "maxsv").

        Returns:
            pd.Series | pd.DataFrame: The computed iterative triangulation.
        """
        # Get timestamps from the navigation file
        nav = self._choose_nav_message_for_interpolation(
            obs,
            nav,
            ephemeris='maxsv' if 'ephemeris' not in kwargs else kwargs['ephemeris'],
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
        coords[['x', 'y', 'z']] = earth_rotation_correction(
            coords[['x', 'y', 'z']].to_numpy(), obs.data['dt'].to_numpy().ravel()
        )

        # Attach the relevant statistics to a new frame that contains
        # pseudorange and sv coordinates
        stats = obs.data[['Pseudorange']].join(coords)

        # TO DO : Now to perform the iterative triangulation
        # using the stats dataframe. Newton's method can be used
        # to solve the observation equations for the receiver position and clock offset.
        print("Under Construction. 🚧🦺  🚧 🦺  🚧")

        return stats
