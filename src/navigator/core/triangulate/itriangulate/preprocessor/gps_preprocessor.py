"""Preprocessor for GPS epoch data."""

import numpy as np
import pandas as pd

from .....epoch.epoch import Epoch
from .....utility.transforms.coordinate_transforms import geocentric_to_ellipsoidal
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
from ..algos.wls.wls_triangulator import wls_triangulation
from .preprocessor import Preprocessor

__all__ = ["GPSPreprocessor"]


class GPSPreprocessor(Preprocessor):
    """Preprocessor for GPS epoch data.

    Args:
        Preprocessor (_type_): Abstract class for a data preprocessor.
    """

    L1_CODE_ON = Epoch.L1_CODE_ON
    L2_CODE_ON = Epoch.L2_CODE_ON
    L1_PHASE_ON = Epoch.L1_PHASE_ON
    L2_PHASE_ON = Epoch.L2_PHASE_ON

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
        rotated = earth_rotation_correction(
            sv_position=sv_coords[["x", "y", "z"]].to_numpy(dtype=np.float64),
            dt=dt.to_numpy(dtype=np.float64).ravel(),
        )

        return pd.DataFrame(
            data=rotated,
            index=sv_coords.index,
            columns=["x", "y", "z"],
        )

    def process_sv_coordinate(
        self,
        reception_time: pd.Timestamp,
        navigation_data: pd.DataFrame,
        navigation_metadata: pd.Series,
        pseudorange: pd.Series,
        approx_receiver_location: pd.Series | None = None,
        no_clock_correction: bool = False,
    ) -> pd.DataFrame:
        """Process the satellite coordinates at the reception epoch.

        Args:
            reception_time (pd.Timestamp): Reception time of the GPS observations.
            navigation_data (pd.DataFrame): Navigation data.
            navigation_metadata (pd.Series): Metadata for the navigation data.
            pseudorange (pd.Series): Pseudorange of the GPS observations.
            approx_receiver_location (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.
            no_clock_correction (bool, optional): If True, then no satellite clock correction is applied. Defaults to False.

        Returns:
            pd.DataFrame: The processed satellite coordinates at the reception epoch.
        """
        # Compute the satellite coordinates at the emission epoch
        coords = self._compute_sv_coordinates_at_emission_epoch(
            reception_time=reception_time,
            pseudorange=pseudorange,
            nav=navigation_data,
            nav_metadata=navigation_metadata,
            no_clock_correction=no_clock_correction,
        )

        # Rotate the satellite coordinates to the reception epoch
        rotated_coords = self._rotate_satellite_coordinates_to_reception_epoch(
            sv_coords=coords,
            pseudorange=pseudorange,
            approx_receiver_location=approx_receiver_location,
        )

        # Update the satellite coordinates
        coords[["x", "y", "z"]] = rotated_coords

        return coords

    def bootstrap(
        self,
        epoch: Epoch,
    ) -> pd.Series:
        """Bootstrap the receiver location using the WLS.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.

        Returns:
            pd.Series: The approximate receiver location in ECEF coordinate.
        """
        # Process the satellite coordinates at the reception epoch
        coords = self.process_sv_coordinate(
            reception_time=epoch.timestamp,
            navigation_data=epoch.nav_data,
            navigation_metadata=epoch.nav_meta,
            pseudorange=epoch.obs_data[self.L1_CODE_ON],
            approx_receiver_location=None,
            no_clock_correction=False,
        )

        # Get the base solution using WLS
        base_solution = wls_triangulation(
            pseudorange=epoch.obs_data[self.L1_CODE_ON].to_numpy(),
            sv_pos=coords[["x", "y", "z"]].to_numpy(),
            x0=np.zeros(4),
        )
        # Convert the base solution to a pandas series
        solution = pd.Series(
            {
                "x": base_solution["solution"][0],
                "y": base_solution["solution"][1],
                "z": base_solution["solution"][2],
                "cdt": base_solution["solution"][3],
            }
        )
        # Compute the latitude, longitude and height of the receiver based on the base solution
        lat, lon, height = geocentric_to_ellipsoidal(
            x=solution["x"],
            y=solution["y"],
            z=solution["z"],
            max_iter=1000,
        )

        # Attach the latitude, longitude and height to the solution
        solution["lat"] = lat
        solution["lon"] = lon
        solution["height"] = height

        return solution

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

    def preprocess(
        self,
        epoch: Epoch,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess the observation and navigation data to be used for triangulation!

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Pseduorange and satellite coordinates and intermediate results at the common reception epoch.
        """
        # Check for the L1 observation data
        if (
            self.L1_CODE_ON not in epoch.obs_data.columns
            or self.L1_PHASE_ON not in epoch.obs_data.columns
        ):
            raise ValueError(
                f"Invalid observation data. Must contain L1 Code({self.L1_CODE_ON}) and L1 Phase observations({self.L1_PHASE_ON}) to perform processing."
            )

        # Get the prior receiver location if available or use base WLS to solve for it
        # without the need for the prior receiver location
        # This will result in degraded accuracy but it is good enough for applying the corrections
        if epoch.approximate_coords.empty or any(
            [
                val in epoch.approximate_coords
                for val in ["x", "y", "z", "lat", "lon", "height"]
            ]
        ):
            # Get the bootstrap receiver location
            epoch.approximate_coords = self.bootstrap(epoch)

        # Get the satellite coordinates at the emission epoch
        coords = self.process_sv_coordinate(
            reception_time=epoch.timestamp,
            navigation_data=epoch.nav_data,
            navigation_metadata=epoch.nav_meta,
            pseudorange=epoch.obs_data[self.L1_CODE_ON],
            approx_receiver_location=epoch.approximate_coords,
            no_clock_correction=kwargs.get("no_clock_correction", False),
        )

        # Get the elevation and azimuth of the satellites and attach it to the coords
        coords["elevation"], coords["azimuth"] = (
            self._compute_satellite_azimuth_and_elevation(
                sv_coords=coords, approx_receiver_location=epoch.approximate_coords
            )
        )

        # If the epoch is smoothed, then apply swap the smoothed key as C1C
        if epoch.is_smoothed:
            if BaseSmoother.SMOOOTHING_KEY in epoch.obs_data.columns:
                epoch.obs_data["C1C"] = epoch.obs_data[
                    BaseSmoother.SMOOOTHING_KEY
                ]  # Swap the smoothed key as C1C

        # Extract the keyword arguments
        if kwargs.get("verbose", False):
            print(f"Epoch: {epoch.timestamp}")
            print(f"Mode: {epoch.profile.get('mode', 'single')}")
            print(
                f"Apply Ionosphere Correction: {epoch.profile.get('apply_iono', False)}"
            )
            print(
                f"Apply Troposphere Correction: {epoch.profile.get('apply_tropo', False)}"
            )

        # Check if the mode is dual or single and process the observations accordingly
        if epoch.profile.get("mode", "single") == "dual":
            # Check if the L1 and L2 code observations are available
            if (
                self.L1_CODE_ON not in epoch.obs_data.columns
                or self.L2_CODE_ON not in epoch.obs_data.columns
            ):
                raise ValueError(
                    f"Invalid observation data. Must contain both L1 Code({self.L1_CODE_ON}) and L2 Code observations({self.L2_CODE_ON}) to perform dual mode processing."
                )
            # Compute the ionospheric free combination
            pseudorange = pd.Series(
                ionosphere_free_combination(
                    p1=epoch.obs_data[self.L1_CODE_ON].to_numpy(dtype=np.float64),
                    p2=epoch.obs_data[self.L2_CODE_ON].to_numpy(dtype=np.float64),
                ),
                index=epoch.obs_data.index,
                name=self.L1_CODE_ON,
            )

            # Check if the L1 and L2 phase observations are available
            if (
                self.L1_PHASE_ON not in epoch.obs_data.columns
                or self.L2_PHASE_ON not in epoch.obs_data.columns
            ):
                raise ValueError(
                    f"Invalid observation data. Must contain both L1 Phase({self.L1_PHASE_ON}) and L2 Phase observations({self.L2_PHASE_ON}) to perform dual mode processing."
                )

            # Compute the ionospheric free combination for the phase measurements
            phase = pd.Series(
                ionosphere_free_combination(
                    p1=epoch.obs_data[self.L1_PHASE_ON].to_numpy(dtype=np.float64)
                    * L1_WAVELENGTH,  # Scale the L1 phase to meters
                    p2=epoch.obs_data[self.L2_PHASE_ON].to_numpy(dtype=np.float64)
                    * L2_WAVELENGTH,  # Scale the L2 phase to meters
                ),
                index=epoch.obs_data.index,
                name=self.L1_PHASE_ON,
            )

        elif epoch.profile.get("mode", "single") == "single":
            # Get the pseudorange
            pseudorange = epoch.obs_data[self.L1_CODE_ON]
            phase = epoch.obs_data[self.L1_PHASE_ON] * L1_WAVELENGTH

        else:
            raise ValueError(
                f"Invalid mode flag: {epoch.profile.get('mode', 'single')}. Supported modes are ['dual', 'single']"
            )

        # Apply Troposheric Correction
        if epoch.profile.get("apply_tropo", False):
            coords["tropospheric_correction"] = self._compute_tropospheric_correction(
                day_of_year=epoch.timestamp.dayofyear,
                sv_coords=coords,
                approx_receiver_location=epoch.approximate_coords,
            )
            # Correct the pseudorange for the tropospheric correction
            pseudorange -= coords["tropospheric_correction"]
            phase -= coords["tropospheric_correction"]

        if epoch.profile.get("apply_iono", False) and epoch.profile.get("mode", "dual") == "single": # Only apply ionospheric correction for single mode
            # Check if the ionospheric correction parameters are available
            if epoch.nav_meta.get("IONOSPHERIC CORR", None) is None:
                raise ValueError(
                    "Invalid ionospheric correction parameters. Must contain ['IONOSPHERIC CORR'] in the navigation metadata."
                )

            coords["ionospheric_correction"] = self._compute_ionospheric_correction(
                sv_coords=coords,
                approx_receiver_location=epoch.approximate_coords,
                time=epoch.timestamp,
                iono_params=epoch.nav_meta.get("IONOSPHERIC CORR", None),
            )
            # Apply the ionospheric correction
            pseudorange -= coords["ionospheric_correction"]
            phase -= coords["ionospheric_correction"]

        # Apply the satelllite color correction to the pseudorange and phase measurements
        # The clock bias is calculated as the product of the satellite clock bias and the speed of light
        pseudorange += SPEED_OF_LIGHT * coords["dt"]
        phase += SPEED_OF_LIGHT * coords["dt"]

        return pd.concat([pseudorange, phase], axis=1), coords

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
