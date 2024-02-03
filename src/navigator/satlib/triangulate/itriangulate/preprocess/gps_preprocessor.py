"""Preprocessor for GPS epoch data."""

from warnings import warn

import numpy as np
import pandas as pd

from .....utility.epoch.epoch import Epoch
from ....satellite.iephm.sv.igps_ephm import IGPSEphemeris
from ....satellite.iephm.sv.tools.elevation_and_azimuthal import elevation_and_azimuthal
from ....satellite.satellite import Satellite
from ..algos.dual_frequency_corrections import dual_channel_correction
from ..algos.klobuchar_ionospheric_model import klobuchar_ionospheric_correction
from ..algos.rotations import earth_rotation_correction
from ..algos.tropospheric_delay import tropospheric_delay_correction
from .preprocessor import Preprocessor

__all__ = ["GPSPreprocessor"]


class GPSPreprocessor(Preprocessor):
    """Preprocessor for GPS epoch data.

    Args:
        Preprocessor (_type_): Abstract class for a data preprocessor.
    """

    def __init__(self) -> None:
        """Initializes the preprocessor with the GPS constellation."""
        super().__init__(constellation="G")

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

    def _compute_satellite_azimuth_and_elevation(
        self, sv_coords: np.ndarray, approx_receiver_location: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Compute the azimuth and elevation of the satellites.

        Args:
            sv_coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx_receiver_location (pd.Series): Approximate receiver location in ECEF coordinate.

        Returns:
            tuple[pd.Series, pd.Series]: The azimuth and elevation of the satellites.
        """
        # Compute the azimuth and elevation of the satellites
        E, A = elevation_and_azimuthal(
            sv_coords[['x', 'y', 'z']].values,
            approx_receiver_location[['x', 'y', 'z']].values,
        )

        # Attach the azimuth and elevation to the satellite coordinates
        return pd.Series(data=E, index=sv_coords.index, name="elevation"), pd.Series(
            data=A, index=sv_coords.index, name="azimuth"
        )

    def _compute_tropospheric_correction(
        self,
        day_of_year: int,
        sv_coords: pd.DataFrame,
        approx_receiver_location: pd.Series,
        **kwargs,
    ) -> pd.Series:
        """Compute the tropospheric correction for the GPS observations.

        Args:
            day_of_year (int): Day of the year.
            sv_coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx_receiver_location (pd.Series): Approximate receiver location in ECEF coordinate.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: The tropospheric correction for the GPS observations.
        """
        # Compute the tropospheric correction
        corr = []
        for index, row in sv_coords.iterrows():
            corr.append(
                tropospheric_delay_correction(
                    elevation=row["elevation"],
                    height=approx_receiver_location["height"],
                    day_of_year=day_of_year,
                    hemisphere=True
                    if approx_receiver_location["lat"] > 0
                    else False,
                    mapping_function=kwargs.get("mapping_function", "neil"),
                )
            )
        return pd.Series(
            data=corr, index=sv_coords.index, name="tropospheric_correction"
        )

    def _compute_ionospheric_correction(
        self,
        sv_coords: pd.DataFrame,
        approx_receiver_location: pd.Series,
        time: pd.Timestamp,
        iono_params: pd.Series,
        **kwargs,
    ) -> pd.Series:
        """Compute the ionospheric correction for the GPS observations.

        Args:
            sv_coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx_receiver_location (pd.Series): Approximate receiver location in ECEF coordinate.
            time (pd.Timestamp): Reception time of the GPS observations.
            iono_params (pd.Series): Ionospheric parameters for the GPS observations.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: The ionospheric correction for the GPS observations.

        """
        # Calculate the gps in seconds of the week
        time = (time - pd.Timestamp("1980-01-06")).total_seconds() % (
            7 * 24 * 60 * 60
        )

        # Compute the ionospheric correction
        iono_corr = []
        for index, row in sv_coords.iterrows():
            iono_corr.append(
                klobuchar_ionospheric_correction(
                    E=row["elevation"],
                    A=row["azimuth"],
                    ionospheric_parameters=iono_params,
                    latitude=approx_receiver_location["lat"],
                    longitude=approx_receiver_location["lon"],
                    t=time,
                )
            )
     
        return pd.Series(
            data=iono_corr, index=sv_coords.index, name="ionospheric_correction"
        )

    def _mode_processing(
        self,
        time: pd.Timestamp,
        mode: str,
        obs_data: pd.DataFrame,
        coords: pd.DataFrame,
        approx: pd.Series = None,
        nav_metadata: pd.Series = None,
        **kwargs,
    ) -> pd.Series:
        """Process the mode flag for the GPS observations.

        Args:
            time (pd.Timestamp): Reception time of the GPS observations.
            mode (str): Mode flag for the GPS observations ['dual', 'single']
            obs_data (pd.DataFrame): GPS observations data.
            coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.
            nav_metadata (pd.Series, optional): Metadata for the navigation data. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: The processed pseudorange for the GPS observations.
        """
        if mode == "dual":
            # Compute the ionospheric free combination
            pseudorange = self._ionospehric_free_combination(obs_data, code_warnings=kwargs.get("verbose", False))

            if approx is None:
                return pseudorange  # Do not apply the tropospheric correction since the approximate receiver location is not provided

            # Compute the azimuth and elevation of the satellites
            (
                coords["elevation"],
                coords["azimuth"],
            ) = self._compute_satellite_azimuth_and_elevation(
                sv_coords=coords, approx_receiver_location=approx
            )
            
            # Compute the tropospheric correction
            coords["tropospheric_correction"] = self._compute_tropospheric_correction(
                day_of_year=time.dayofyear,
                sv_coords=coords,
                approx_receiver_location=approx,
                **kwargs,
            )

            # Correct the psedurange for the tropospheric correction
            pseudorange -= coords[
                "tropospheric_correction"
            ]  # Apply the tropospheric correction

            return pseudorange

        if mode == "single":
            # Compute the ionospheric free combination
            pseudorange = obs_data["C1C"]

            # Check if the approximate receiver location is provided
            if approx is None:
                warn(
                    """Approximate receiver location not provided in single mode!. Tropospheric correction not applied.
                    Expect degraded accuracy in the computed position."""
                )
                return pseudorange

            # Compute the azimuth and elevation of the satellites
            (
                coords["elevation"],
                coords["azimuth"],
            ) = self._compute_satellite_azimuth_and_elevation(
                sv_coords=coords, approx_receiver_location=approx
            )

            # Compute the tropospheric correction
            coords["tropospheric_correction"] = self._compute_tropospheric_correction(
                day_of_year=time.dayofyear,
                sv_coords=coords,
                approx_receiver_location=approx,
                **kwargs,
            )

            # Correct the psedurange for the tropospheric correction
            pseudorange -= coords["tropospheric_correction"]

            # Compute the ionospheric correction if nav_metadata is provided
            if nav_metadata is not None and "IONOSPHERIC CORR" in nav_metadata:
                # Compute the ionospheric correction
                coords["ionospheric_correction"] = self._compute_ionospheric_correction(
                    sv_coords=coords,
                    approx_receiver_location=approx,
                    time=time,
                    iono_params=nav_metadata["IONOSPHERIC CORR"],
                    **kwargs,
                )

                # Correct the pseudorange for the ionospheric correction
                pseudorange -= coords["ionospheric_correction"]

            else:
                warn(
                    "Ionospheric correction not applied. Navigation metadata not provided or IONOSPHERIC CORR not present."
                )
            return pseudorange

        raise ValueError("Invalid mode flag. Must be either 'dual' or 'single'.")

    def preprocess(
        self,
        obs: Epoch,
        obs_metadata: pd.Series,  # noqa: ARG002
        nav_metadata: pd.Series,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess the observation and navigation data to be used for triangulation!

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.
            obs_metadata (pd.Series): Metadata for the observation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            **kwargs: Additional keyword arguments.


        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Pseduorange and satellite coordinates at the common reception epoch.
        """
        # Use Epoch to get the navigation message for the observation epoch. Held at "Epoch.nav_data" attribute
        obs_data, nav_data = obs.obs_data, obs.nav_data

        # Compute the satellite coordinates at the emission epoch
        # This also computes satellite clock correction which is stored in the 'dt' column.
        coords = self._compute_sv_coordinates_at_emission_epoch(
            reception_time=obs.timestamp,
            pseudorange=obs_data["C1C"],
            nav=nav_data,
            nav_metadata=nav_metadata,
            **kwargs,
        )

        # Need to apply the earth rotation correction since SV coordinates are in ECEF in emission epoch
        # Need to rotate each satellite coordinate to the reception epoch since it is common epoch for all satellites
        coords = self._rotate_satellite_coordinates_to_reception_epoch(
            sv_coords=coords,
            pseudorange=obs_data["C1C"],
            approx_receiver_location=kwargs.get("approx", None),
        )

        # Compute the mode flag for the GPS observations
        mode = kwargs.get("mode", "dual")
        # Remove the mode flag from the kwargs
        kwargs.pop("mode", None)

        # Get the approximate receiver location if provided
        approx = kwargs.get("approx", None)
        kwargs.pop("approx", None)

        # Process the mode flag for the GPS observations
        pseudorange = self._mode_processing(
            time=obs.timestamp,
            mode=mode,
            obs_data=obs_data,
            coords=coords,
            approx=approx,
            nav_metadata=nav_metadata,
            **kwargs
        )
        # Correct the pseudorange for the satellite clock offset.
        # This crossponds to the satellite clock correction. P(j) + c * dt(j)
        pseudorange += coords["dt"] * 299792458

        return pseudorange, coords[['x', 'y', 'z']]

    def __call__(
        self,
        obs: Epoch,
        obs_metadata: pd.Series,  # noqa: ARG002
        nav_metadata: pd.Series,
        **kwargs,
    ) -> None:
        """Preprocess the observation and navigation data to be used for triangulation!

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.
            obs_metadata (pd.Series): Metadata for the observation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            **kwargs: Additional keyword arguments.


        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Pseduorange and satellite coordinates at the common reception epoch.
        """
        return self.preprocess(
            obs=obs,
            obs_metadata=obs_metadata,
            nav_metadata=nav_metadata,
            **kwargs,
        )
