"""Preprocessor for GPS epoch data."""

from warnings import warn

import numpy as np
import pandas as pd

from .....utility.epoch.epoch import Epoch
from ....satellite.iephm.sv.igps_ephm import IGPSEphemeris
from ....satellite.iephm.sv.tools.elevation_and_azimuthal import elevation_and_azimuthal
from ....satellite.satellite import Satellite
from ..algos.ionosphere.dual_frequency_corrections import dual_channel_correction
from ..algos.ionosphere.klobuchar_ionospheric_model import (
    klobuchar_ionospheric_correction,
)
from ..algos.rotations import earth_rotation_correction
from ..algos.troposphere.tropospheric_delay import tropospheric_delay_correction
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
        no_clock_correction: bool = False,
    ) -> Epoch:
        """Computes the satellite coordinates at the emission epoch.

        This method computes the satellite coordinates at the emission epoch using the GPS observations and navigation data.
        It instantiates the Satellite class and computes the satellite coordinate at the emission epoch.

        Args:
            reception_time (pd.Timestamp): Reception time of the GPS observations.
            pseudorange (pd.Series): Pseudorange of the GPS observations.
            nav (pd.DataFrame): Navigation data.
            nav_metadata (pd.Series): Metadata for the navigation data.
            no_clock_correction (bool, optional): If True, then no satellite clock correction is applied. Defaults to False.

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
            t_sv=t_sv,
            metadata=nav_metadata,
            data=nav,
            no_clock_correction=no_clock_correction,
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
    ) -> pd.Series:
        """Compute the tropospheric correction for the GPS observations.

        Args:
            day_of_year (int): Day of the year.
            sv_coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx_receiver_location (pd.Series): Approximate receiver location in ECEF coordinate.

        Returns:
            pd.Series: The tropospheric correction for the GPS observations.
        """
        # Compute the tropospheric correction
        corr = []
        for index, row in sv_coords.iterrows():
            corr.append(
                tropospheric_delay_correction(
                    latitude=approx_receiver_location["lat"],
                    elevation=row["elevation"],
                    height=approx_receiver_location["height"],
                    day_of_year=day_of_year,
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
        time = (time - pd.Timestamp("1980-01-06")).total_seconds() % (7 * 24 * 60 * 60)

        # Compute the ionospheric correction
        iono_corr = []
        for index, row in sv_coords.iterrows():
            iono_corr.append(
                klobuchar_ionospheric_correction(
                    elev=row["elevation"],
                    azimuth=row["azimuth"],
                    latitude=approx_receiver_location["lat"],
                    longitude=approx_receiver_location["lon"],
                    tow=time,
                    alpha0=iono_params["alpha0"],
                    alpha1=iono_params["alpha1"],
                    alpha2=iono_params["alpha2"],
                    alpha3=iono_params["alpha3"],
                    beta0=iono_params["beta0"],
                    beta1=iono_params["beta1"],
                    beta2=iono_params["beta2"],
                    beta3=iono_params["beta3"],
                )
            )

        return pd.Series(
            data=iono_corr, index=sv_coords.index, name="ionospheric_correction"
        )

    def _dual_mode_processing(
        self,
        time: pd.Timestamp,
        obs_data: pd.DataFrame,
        coords: pd.DataFrame,
        approx: pd.Series = None,
        apply_tropo: bool = True,
        apply_iono: bool = True,
        verbose: bool = False,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Dual mode processing for the GPS observations.

        Args:
            time (pd.Timestamp): Reception time of the GPS observations.
            obs_data (pd.DataFrame): GPS observations data.
            coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.
            apply_tropo (bool, optional): If True, then tropospheric correction is applied. Defaults to True.
            apply_iono (bool, optional): If True, then ionospheric correction is applied. Defaults to True.
            verbose (bool, optional): If True, then warning is raised. Defaults to False.

        Returns:
            tuple[pd.Series, pd.DataFrame]: The processed pseudorange and intermediate results.
        """
        # Compute the ionospheric free combination
        pseudorange = (
            self._ionospehric_free_combination(obs_data, code_warnings=verbose)
            if apply_iono
            else obs_data["C1C"]
        )

        if apply_tropo:
            if approx is None:
                raise ValueError(
                    """A priori receiver location not provided in dual mode!. Tropospheric correction cannot be applied.
                    Explictly pass apply_tropo=False to kwargs to proceed without tropospheric correction."""
                )

            # Compute the azimuth and elevation of the satellites
            (
                coords["elevation"],
                coords["azimuth"],
            ) = self._compute_satellite_azimuth_and_elevation(
                sv_coords=coords, approx_receiver_location=approx
            )

            coords["tropospheric_correction"] = self._compute_tropospheric_correction(
                day_of_year=time.dayofyear,
                sv_coords=coords,
                approx_receiver_location=approx,
            )
            # Correct the psedurange for the tropospheric correction
            pseudorange -= coords[
                "tropospheric_correction"
            ]  # Apply the tropospheric correction

        return pseudorange, coords

    def _single_mode_processing(
        self,
        time: pd.Timestamp,
        obs_data: pd.DataFrame,
        coords: pd.DataFrame,
        nav_metadata: pd.Series = None,
        approx: pd.Series = None,
        apply_tropo: bool = True,
        apply_iono: bool = True,
        verbose: bool = False,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Single mode processing for the GPS observations.

        Args:
            time (pd.Timestamp): Reception time of the GPS observations.
            obs_data (pd.DataFrame): GPS observations data.
            coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.
            nav_metadata (pd.Series, optional): Metadata for the navigation data. Defaults to None.
            apply_tropo (bool, optional): If True, then tropospheric correction is applied. Defaults to True.
            apply_iono (bool, optional): If True, then ionospheric correction is applied. Defaults to True.
            verbose (bool, optional): If True, then warning is raised. Defaults to False.

        Returns:
            tuple[pd.Series, pd.DataFrame]: The processed pseudorange and intermediate results.
        """
        # Get the range for the single frequency [C1C]
        pseudorange = obs_data["C1C"]

        # Compute the satellite azimuth and elevation if any of the apply_tropo or apply_iono is True
        if (apply_tropo or apply_iono) and approx is not None:
            # Compute the azimuth and elevation of the satellites
            (
                coords["elevation"],
                coords["azimuth"],
            ) = self._compute_satellite_azimuth_and_elevation(
                sv_coords=coords, approx_receiver_location=approx
            )
        else:
            if verbose:
                warn(
                    "Tropospheric and Ionospheric correction are not applied. Expect degeraded accuracy!."
                )

        # Apply the tropospheric correction if the approximate receiver location is provided
        if apply_tropo:
            if approx is None:
                raise ValueError(
                    """Approximate receiver location not provided in single mode!. Tropospheric correction cannot be applied.
                    Explictly pass apply_tropo=False to kwargs to proceed without tropospheric correction."""
                )
            # TROPOSPHERIC CORRECTION
            # Compute the tropospheric correction

            coords["tropospheric_correction"] = self._compute_tropospheric_correction(
                day_of_year=time.dayofyear,
                sv_coords=coords,
                approx_receiver_location=approx,
            )
            # Correct the psedurange for the tropospheric correction
            pseudorange -= coords[
                "tropospheric_correction"
            ]  # Apply the tropospheric correction

        # Apply the ionospheric correction if the ionospheric parameters are provided
        if apply_iono:
            if nav_metadata is None or "IONOSPHERIC CORR" not in nav_metadata:
                raise ValueError(
                    """Ionospheric parameters not provided in single mode!. Ionospheric correction cannot be applied.
                    Explictly pass apply_iono=False to kwargs to proceed without ionospheric correction."""
                )
            # IONOSPHERIC CORRECTION
            # Compute the ionospheric correction
            coords["ionospheric_correction"] = self._compute_ionospheric_correction(
                sv_coords=coords,
                approx_receiver_location=approx,
                time=time,
                iono_params=nav_metadata["IONOSPHERIC CORR"],
            )

            # Apply the ionospheric correction
            pseudorange -= coords[
                "ionospheric_correction"
            ]  # Apply the ionospheric correction

        return pseudorange, coords

    def _dispatch_mode(
        self,
        time: pd.Timestamp,
        mode: str,
        obs_data: pd.DataFrame,
        coords: pd.DataFrame,
        approx: pd.Series = None,
        nav_metadata: pd.Series = None,
        apply_tropo: bool = True,
        apply_iono: bool = True,
        verbose: bool = False,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Process the mode flag for the GPS observations.

        Args:
            time (pd.Timestamp): Reception time of the GPS observations.
            mode (str): Mode flag for the GPS observations ['dual', 'single']
            obs_data (pd.DataFrame): GPS observations data.
            coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.
            nav_metadata (pd.Series, optional): Metadata for the navigation data. Defaults to None.
            apply_tropo (bool, optional): If True, then tropospheric correction is applied. Defaults to True.
            apply_iono (bool, optional): If True, then ionospheric correction is applied. Defaults to True.
            verbose (bool, optional): If True, then warning is raised. Defaults to False.

        Returns:
            tuple[pd.Series, pd.DataFrame]: The processed pseudorange and intermediate results.
        """
        if mode.lower() == "dual":
            return self._dual_mode_processing(
                time=time,
                obs_data=obs_data,
                coords=coords,
                approx=approx,
                apply_iono=apply_iono,
                apply_tropo=apply_tropo,
                verbose=verbose,
            )
        if mode.lower() == "single":
            return self._single_mode_processing(
                time=time,
                obs_data=obs_data,
                coords=coords,
                approx=approx,
                nav_metadata=nav_metadata,
                apply_iono=apply_iono,
                apply_tropo=apply_tropo,
                verbose=verbose,
            )

        raise ValueError(
            f"Invalid mode flag: {mode}. Supported modes are ['dual', 'single']."
        )

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

        Additional Keyword Arguments:
            mode (str, optional): Mode flag for the GPS observations ['dual', 'single']. Defaults to "dual".
            prior (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.
            apply_tropo (bool, optional): If True, then tropospheric correction is applied. Defaults to True.
            apply_iono (bool, optional): If True, then ionospheric correction is applied. Defaults to True.
            verbose (bool, optional): If True, then warning is raised. Defaults to False.

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
            no_clock_correction=kwargs.get("no_clock_correction", False),
        )

        # Need to apply the earth rotation correction since SV coordinates are in ECEF in emission epoch
        # Need to rotate each satellite coordinate to the reception epoch since it is common epoch for all satellites
        coords = self._rotate_satellite_coordinates_to_reception_epoch(
            sv_coords=coords,
            pseudorange=obs_data["C1C"],
            approx_receiver_location=kwargs.get("approx", None),
        )

        # Get Necessary parameters from the kwargs
        mode = kwargs.get("mode", "dual")
        approx = kwargs.get("prior", None)
        apply_tropo = kwargs.get("apply_tropo", True)
        apply_iono = kwargs.get("apply_iono", True)
        verbose = kwargs.get("verbose", False)

        # Process the mode flag for the GPS observations
        pseudorange, coords = self._dispatch_mode(
            time=obs.timestamp,
            mode=mode,
            obs_data=obs_data,
            coords=coords,
            approx=approx,
            nav_metadata=nav_metadata,
            apply_iono=apply_iono,
            apply_tropo=apply_tropo,
            verbose=verbose,
        )

        # Correct the pseudorange for the satellite clock offset.
        # This crossponds to the satellite clock correction. P(j) + c * dt(j)
        pseudorange += coords["dt"] * 299792458

        return pseudorange, coords

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
            tuple[pd.DataFrame, pd.DataFrame]: Pseduorange and satellite coordinates and intermediate results at the common reception epoch.
        """
        return self.preprocess(
            obs=obs,
            obs_metadata=obs_metadata,
            nav_metadata=nav_metadata,
            **kwargs,
        )
