"""Preprocessor for GPS epoch data."""

from warnings import warn

import numpy as np
import pandas as pd

from .....epoch.epoch import Epoch
from ....satellite.iephm.sv.igps_ephm import IGPSEphemeris
from ....satellite.iephm.sv.tools.elevation_and_azimuthal import (
    elevation_and_azimuthal,
)
from ....satellite.satellite import Satellite
from ..algos.combinations.range_combinations import (
    L1_WAVELENGTH,
    L2_WAVELENGTH,
    SPEED_OF_LIGHT,
    ionosphere_free_combination,
)
from ..algos.ionosphere.klobuchar_ionospheric_model import (
    klobuchar_ionospheric_correction,
)
from ..algos.rotations import earth_rotation_correction
from ..algos.smoothing.base_smoother import BaseSmoother
from ..algos.troposphere.tropospheric_delay import tropospheric_delay_correction
from .preprocessor import Preprocessor

__all__ = ["GPSPreprocessor"]


class GPSPreprocessor(Preprocessor):
    """Preprocessor for GPS epoch data.

    Args:
        Preprocessor (_type_): Abstract class for a data preprocessor.
    """

    L1_CODE_ON = "C1C"
    L2_CODE_ON = "C2W"
    L1_PHASE_ON = "L1C"
    L2_PHASE_ON = "L2W"

    PRIOR_KEY = "prior"

    def __init__(self) -> None:
        """Initializes the preprocessor with the GPS constellation."""
        super().__init__(constellation="G")

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
        dt = pseudorange / SPEED_OF_LIGHT

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
                [coord in approx_receiver_location.index for coord in ["x", "y", "z"]]
            ):
                raise ValueError(
                    "Invalid approx_receiver_location. Must contain ['x', 'y', 'z'] coordinates."
                )
            # Compute the dt using method in Equation 5.13 in ESA GNSS Book
            # https://gssc.esa.int/navipedia/GNSS_Book/ESA_GNSS-Book_TM-23_Vol_I.pdf
            dt = (
                sv_coords[["x", "y", "z"]] - approx_receiver_location[["x", "y", "z"]]
            ).apply(lambda row: row.dot(row) ** 0.5 / 299792458, axis=1)

        # Rotate the satellite coordinates to the reception epoch
        sv_coords[["x", "y", "z"]] = earth_rotation_correction(
            sv_position=sv_coords[["x", "y", "z"]].to_numpy(dtype=np.float64),
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
            sv_coords[["x", "y", "z"]].values,
            approx_receiver_location[["x", "y", "z"]].values,
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
        t = (time - pd.Timestamp("1980-01-06")).total_seconds() % (7 * 24 * 60 * 60)

        # Compute the ionospheric correction
        iono_corr = []
        for index, row in sv_coords.iterrows():
            iono_corr.append(
                klobuchar_ionospheric_correction(
                    elev=row["elevation"],
                    azimuth=row["azimuth"],
                    latitude=approx_receiver_location["lat"],
                    longitude=approx_receiver_location["lon"],
                    tow=t,
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

    def _check_keys(self, data: pd.DataFrame, key: list[str]) -> None:
        """Check if the required keys are present in the data.

        Args:
            data (pd.DataFrame): Data to be checked.
            key (list[str]): List of keys to be checked.

        Raises:
            ValueError: If any of the keys is not present in the data.
        """
        for k in key:
            if k not in data.columns:
                raise KeyError(f"Key {k} not found in the data.")

    def _dual_mode_processing(
        self,
        time: pd.Timestamp,
        obs_data: pd.DataFrame,
        coords: pd.DataFrame,
        approx: pd.Series = None,
        apply_tropo: bool = True,
        apply_iono: bool = True,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Dual mode processing for the GPS observations.

        Args:
            time (pd.Timestamp): Reception time of the GPS observations.
            obs_data (pd.DataFrame): GPS observations data.
            coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.
            apply_tropo (bool, optional): If True, then tropospheric correction is applied. Defaults to True.
            apply_iono (bool, optional): If True, then ionospheric correction is applied. Defaults to True.

        Returns:
            tuple[pd.Series, pd.DataFrame]: The processed pseudorange and intermediate results.
        """
        # Check if all the data is available
        try:
            self._check_keys(data=obs_data, key=[self.L1_CODE_ON, self.L2_CODE_ON])
        except KeyError:
            raise ValueError(
                f"Invalid observation data. Must contain both L1 Code({self.L1_CODE_ON}) and L2 Code observations({self.L2_CODE_ON}) to perform dual mode processing."
            )

        # Compute the ionospheric free combination
        pseudorange = (
            pd.Series(
                ionosphere_free_combination(
                    p1=obs_data[self.L1_CODE_ON].to_numpy(dtype=np.float64),
                    p2=obs_data[self.L2_CODE_ON].to_numpy(dtype=np.float64),
                ),
                name="code",
                index=obs_data.index,
            )
            if apply_iono
            else obs_data[
                self.L1_CODE_ON
            ]  # Do not apply ionospheric correction if explicitly set to False
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
        # Check if the L1 code is available
        try:
            self._check_keys(data=obs_data, key=[self.L1_CODE_ON])
        except KeyError:
            raise ValueError(
                f"Invalid observation data. Must contain L1 Code observations({self.L1_CODE_ON}) to perform single mode processing"
            )

        # Get the pseudorange
        pseudorange = obs_data[self.L1_CODE_ON]

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

    def _phase_kalman_filter_processing(
        self,
        time: pd.Timestamp,
        obs_data: pd.DataFrame,
        coords: pd.DataFrame,
        approx: pd.Series = None,
        apply_tropo: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process the mode flag for the GPS observations.

        Args:
            time (pd.Timestamp): Reception time of the GPS observations.
            mode (str): Mode flag for the GPS observations ['dual', 'single']
            obs_data (pd.DataFrame): GPS observations data.
            coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.
            apply_tropo (bool, optional): If True, then tropospheric correction is applied. Defaults to True.

        Returns:
            tuple[pd.Series, pd.DataFrame]: The processed pseudorange and intermediate results.
        """
        # Check if all the data is available
        try:
            self._check_keys(
                data=obs_data,
                key=[
                    self.L1_PHASE_ON,
                    self.L2_PHASE_ON,
                    self.L1_CODE_ON,
                    self.L2_CODE_ON,
                ],
            )
        except KeyError:
            raise ValueError(
                f"""Invalid observation data. Must contain both L1 Code({self.L1_CODE_ON}) and L2 Code observations({self.L2_CODE_ON}) 
                and L1 Phase({self.L1_PHASE_ON}) and L2 Phase({self.L2_PHASE_ON}) to perform phase based processing."""
            )

        # Approx is needed for the phase based measurements
        if approx is None:
            raise ValueError(
                """Approximate receiver location not provided in phase based processing!. Phase based measurements cannot be computed.
                To avoid this,use WLS with initial epoch profile to get initial receiver location."""
            )

        # Scale the phase measurements to the meters from cycles
        obs_data[self.L1_PHASE_ON] *= L1_WAVELENGTH
        obs_data[self.L2_PHASE_ON] *= L2_WAVELENGTH

        # Compute the ionospheric free combination
        phase_if = pd.Series(
            ionosphere_free_combination(
                p1=obs_data[self.L1_PHASE_ON].to_numpy(),
                p2=obs_data[self.L2_PHASE_ON].to_numpy(),
            ),
            index=obs_data.index,
            name="phase",
        )
        code_if = pd.Series(
            ionosphere_free_combination(
                p1=obs_data[self.L1_CODE_ON].to_numpy(),
                p2=obs_data[self.L2_CODE_ON].to_numpy(),
            ),
            index=obs_data.index,
            name="code",
        )

        # Kalman Filter needs the elevation and azimuth of satellites
        # Hence compute the azimuth and elevation of the satellites
        (
            coords["elevation"],
            coords["azimuth"],
        ) = self._compute_satellite_azimuth_and_elevation(
            sv_coords=coords, approx_receiver_location=approx
        )

        # Apply the tropospheric correction if required
        if apply_tropo:
            coords["tropospheric_correction"] = self._compute_tropospheric_correction(
                day_of_year=time.dayofyear,
                sv_coords=coords,
                approx_receiver_location=approx,
            )
            code_if -= coords["tropospheric_correction"]
            phase_if -= coords["tropospheric_correction"]

        return pd.concat([code_if, phase_if], axis=1), coords

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
    ) -> tuple[pd.Series | pd.DataFrame, pd.DataFrame]:
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
            tuple[pd.Series | pd.Dataframe, pd.DataFrame]: The processed pseudorange and intermediate results.
        """
        if mode.lower() == "dual":
            return self._dual_mode_processing(
                time=time,
                obs_data=obs_data,
                coords=coords,
                approx=approx,
                apply_iono=apply_iono,
                apply_tropo=apply_tropo,
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
        if mode.lower() == "phase":
            return self._phase_kalman_filter_processing(
                time=time,
                obs_data=obs_data,
                coords=coords,
                approx=approx,
                apply_tropo=apply_tropo,
            )
        raise ValueError(
            f"Invalid mode flag: {mode}. Supported modes are ['dual', 'single', 'phase']"
        )

    def preprocess(
        self,
        epoch: Epoch,
        **kwargs,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Preprocess the observation and navigation data to be used for triangulation!

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            **kwargs: Additional keyword arguments.

        Additional Keyword Arguments:
            prior (pd.Series, optional): Prior receiver location in ECEF coordinate. Defaults to None.


        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Pseduorange and satellite coordinates at the common reception epoch.
        """
        # Use Epoch to get the navigation message for the observation epoch. Held at "Epoch.nav_data" attribute
        obs_data, nav_data = epoch.obs_data, epoch.nav_data

        # Compute the satellite coordinates at the emission epoch
        # This also computes satellite clock correction which is stored in the 'dt' column.
        coords = self._compute_sv_coordinates_at_emission_epoch(
            reception_time=epoch.timestamp,
            pseudorange=obs_data[self.L1_CODE_ON],  # Use the L1 code pseudorange
            nav=nav_data,
            nav_metadata=epoch.nav_meta,
            no_clock_correction=kwargs.get("no_clock_correction", False),
        )

        # Need to apply the earth rotation correction since SV coordinates are in ECEF in emission epoch
        # Need to rotate each satellite coordinate to the reception epoch since it is common epoch for all satellites
        coords = self._rotate_satellite_coordinates_to_reception_epoch(
            sv_coords=coords,
            pseudorange=obs_data[self.L1_CODE_ON],  # Use the L1 code pseudorange
            approx_receiver_location=kwargs.get(self.PRIOR_KEY, None),
        )

        # Get the kwargs
        approx = kwargs.get(self.PRIOR_KEY, None)
        verbose = kwargs.get("verbose", False)

        # If the epoch is smoothed, then apply swap the smoothed key as C1C
        if epoch.is_smoothed:
            if BaseSmoother.SMOOOTHING_KEY in obs_data.columns:
                obs_data["C1C"] = obs_data[
                    BaseSmoother.SMOOOTHING_KEY
                ]  # Swap the smoothed key as C1C

        # Process the mode flag for the GPS observations
        pseudorange, coords = self._dispatch_mode(
            time=epoch.timestamp,
            mode=epoch.profile.get("mode", "single"),
            obs_data=obs_data,
            coords=coords,
            approx=approx,
            nav_metadata=epoch.nav_meta,
            apply_iono=epoch.profile.get("apply_iono", False),
            apply_tropo=epoch.profile.get("apply_tropo", False),
            verbose=verbose,
        )

        # hence it needs to be corrected for both
        if isinstance(pseudorange, pd.DataFrame):
            # Correct code
            pseudorange["code"] += coords["dt"] * SPEED_OF_LIGHT
            # Correct phase
            pseudorange["phase"] += coords["dt"] * SPEED_OF_LIGHT

            # # Stack the code and phase measurements in a single pandas series
            # # So that it can be used for triangulation since the triangulation
            # # uses pd.Series to compute the position
            pseudorange = pd.concat([pseudorange["code"], pseudorange["phase"]], axis=0)
        else:
            # Correct the pseudorange for the satellite clock correction
            pseudorange += coords["dt"] * SPEED_OF_LIGHT

        return pseudorange, coords

    def __call__(
        self,
        obs: Epoch,
        **kwargs,
    ) -> None:
        """Preprocess the observation and navigation data to be used for triangulation!

        Args:
            obs (Epoch): Epoch containing observation data and navigation data.
            **kwargs: Additional keyword arguments.


        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Pseduorange and satellite coordinates and intermediate results at the common reception epoch.
        """
        return self.preprocess(
            epoch=obs,
            **kwargs,
        )
