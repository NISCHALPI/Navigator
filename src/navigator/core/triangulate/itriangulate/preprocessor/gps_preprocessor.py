"""Preprocessor for GPS epoch data."""

import numpy as np
import pandas as pd

from .....epoch.epoch import Epoch
from .....utils.transforms.coordinate_transforms import geocentric_to_ellipsoidal
from ....satellite.iephm import IGPSEphemeris, IGPSSp3
from ....satellite.iephm.sv.tools.elevation_and_azimuthal import (
    elevation_and_azimuthal,
)
from ..algos.combinations.range_combinations import (
    L1_WAVELENGTH,
    L2_WAVELENGTH,
    SPEED_OF_LIGHT,
    ionosphere_free_combination,
)
from ..algos.ionosphere.klobuchar_ionospheric_model import (
    klobuchar_ionospheric_correction,
)
from ..algos.rotations import sagnac_correction
from ..algos.smoothing.base_smoother import BaseSmoother
from ..algos.troposphere.tropospheric_delay import (
    saastamoinen_tropospheric_correction_with_neil_mapping,
    unb3m,
)
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

    # Satellite Coord Correction Iterations
    SV_COORD_CORRECTION_ITER = 2

    # Create a GPS satellite ephemeris processor
    ephemeris_processor = IGPSEphemeris()
    sp3_processor = IGPSSp3()

    def __init__(self) -> None:
        """Initializes the preprocessor with the GPS constellation."""
        super().__init__(constellation="G")

    def compute_tropospheric_correction(
        self,
        day_of_year: int,
        sv_coords: pd.DataFrame,
        approx_receiver_location: pd.Series,
        model: str = "saastamoinen",
    ) -> np.ndarray:
        """Compute the tropospheric correction for the GPS observations.

        Args:
            day_of_year (int): Day of the year.
            sv_coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx_receiver_location (pd.Series): Approximate receiver location in ECEF coordinate.
            model (str, optional): The tropospheric model to use. Defaults to "saastamoinen".

        Returns:
            np.ndarray: The tropospheric correction for the GPS observations.
        """
        # Compute the tropospheric correction
        if model == "saastamoinen":
            return sv_coords.apply(
                lambda row: saastamoinen_tropospheric_correction_with_neil_mapping(
                    latitude_of_receiver=approx_receiver_location["lat"],
                    elevation_angle_of_satellite=row["elevation"],
                    height_of_receiver=approx_receiver_location["height"],
                    day_of_year=day_of_year,
                ),
                axis=1,
            ).to_numpy(dtype=np.float64)
        elif model == "unb3m":  # noqa
            return sv_coords.apply(
                lambda row: unb3m(
                    latitude_of_receiver=approx_receiver_location["lat"],
                    elevation_angle_of_satellite=row["elevation"],
                    height_of_receiver=approx_receiver_location["height"],
                    day_of_year=day_of_year,
                ),
                axis=1,
            ).to_numpy(dtype=np.float64)

        else:
            raise ValueError(
                f"Invalid tropospheric model: {model}. Supported models are ['saastamoinen', 'unb3m']"
            )

    def compute_ionospheric_correction(
        self,
        sv_coords: pd.DataFrame,
        approx_receiver_location: pd.Series,
        time: pd.Timestamp,
        iono_params: pd.Series,
    ) -> np.ndarray:
        """Compute the ionospheric correction for the GPS observations.

        Args:
            sv_coords (pd.DataFrame): Satellite coordinates at the reception epoch.
            approx_receiver_location (pd.Series): Approximate receiver location in ECEF coordinate.
            time (pd.Timestamp): Reception time of the GPS observations.
            iono_params (pd.Series): Ionospheric parameters for the GPS observations.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The ionospheric correction for the GPS observations.

        """
        # Calculate the gps in seconds of the week
        t = (time - pd.Timestamp("1980-01-06")).total_seconds() % (7 * 24 * 60 * 60)

        # Compute the ionospheric correction using vectorized operations
        return sv_coords.apply(
            lambda row: klobuchar_ionospheric_correction(
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
            ),
            axis=1,
        ).to_numpy(dtype=np.float64)

    def travel_time(
        self,
        pseudorange: np.ndarray,
        approximate_sv_coords: np.ndarray | None = None,
        approx_receiver_location: np.ndarray | None = None,
    ) -> np.ndarray:
        """Approximate the travel time of the GPS signal.

        To compute the travel time of the GPS signal, two methods are used:
            - If the satellite coordinates and approximate reciever location are available, then the travel time is computed as the norm of the difference between the satellite and receiver coordinates.
            - If the satellite coordinates are not available, then the travel time is computed as the ratio of the pseudorange and the speed of light.

        Args:
            pseudorange (float): The pseudorange of the GPS signal. (N,)
            approximate_sv_coords (np.ndarray, optional): The approximate satellite coordinates in ECEF coordinate. Defaults to None. (N, 3)
            approx_receiver_location (np.ndarray, optional): The approximate receiver location in ECEF coordinate. Defaults to None. (3,)

        Returns:
            np.ndarray: The approximate travel time of the GPS signal. (N,)

        """
        # # Check if approx_receiver_location is None
        if approx_receiver_location is None or approximate_sv_coords is None:
            return pseudorange / SPEED_OF_LIGHT

        # Ensure that the approx_receiver_location contains the ['x', 'y', 'z'] coordinates
        return (
            np.linalg.norm(approximate_sv_coords - approx_receiver_location, axis=1)
            / SPEED_OF_LIGHT
        )

    def compute_sv_coordinates_at_emission_epoch(
        self,
        reception_time: pd.Timestamp,
        pseudorange: pd.Series,
        nav: pd.DataFrame,
        nav_metadata: pd.Series,
        approx_receiver_location: pd.Series | None = None,
        approx_sv_location: pd.DataFrame | None = None,
        is_sp3: bool = False,
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
            approx_receiver_location (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.
            approx_sv_location (pd.DataFrame, optional): Approximate satellite location in ECEF coordinate. Defaults to None.
            is_sp3 (bool, optional): If True, then the navigation data is in SP3 format. Defaults to False.

        Returns:
            Epoch: The computed satellite coordinates at the emission epoch.


        Note:
            - The approximate receiver location should be in ECEF coordinate and have the only keys ['x', 'y', 'z'].
            - The approximate satellite location should be in ECEF coordinate and have the only keys ['x', 'y', 'z'].
        """
        # Get the approximate values of the receiver location and satellite location
        approx_rcvr, rcvr_clock_bias = (
            (
                approx_receiver_location[["x", "y", "z"]].to_numpy(dtype=np.float64),
                approx_receiver_location["cdt"] / SPEED_OF_LIGHT,
            )
            if approx_receiver_location is not None
            else (None, 0.0)
        )
        # Get the approximate values of the satellite location
        approx_sv = (
            approx_sv_location.to_numpy(dtype=np.float64)
            if approx_sv_location is not None
            else None
        )

        # If the data format is ephemeris, then compute the satellite coordinates at the emission epoch
        # Compute the travel time of the GPS signal
        travel_time = self.travel_time(
            pseudorange=(
                pseudorange.to_numpy(dtype=np.float64)
                - SPEED_OF_LIGHT * rcvr_clock_bias
            ),  # Correct for the receiver clock bias if available
            approximate_sv_coords=approx_sv,
            approx_receiver_location=approx_rcvr,
        )

        # Convert to time delta
        emission_epoch = pd.Series(
            reception_time
            - pd.to_timedelta(travel_time, unit="s")  # Correct for the travel time
            - pd.to_timedelta(
                rcvr_clock_bias, unit="s"
            ),  # Correct for the receiver clock bias
            index=pseudorange.index.get_level_values("sv"),
        )

        # If sp3 data is available, then compute the satellite coordinates at the emission epoch
        # using the sp3
        if is_sp3:
            sv_coords = pd.DataFrame(
                [
                    self.sp3_processor.compute(
                        t=emission_epoch[sv],
                        metadata=None,
                        data=nav.xs(key=sv, level="sv"),
                        tolerance=pd.Timedelta(hours=2),
                    )
                    for sv in pseudorange.index
                ],
                index=pseudorange.index.get_level_values("sv"),
            )
        else:
            # If the data format is ephemeris, then compute the satellite coordinates at the emission epoch
            # using the ephemeris data

            # Iterate over the navigation data and compute the satellite coordinates at the emission epoch
            sv_pos = []
            for (toc, sv), ephemeris in nav.iterrows():
                # Add the ephemeris time of clock to the ephemeris data
                # This is due to data formatting # TODO: Fix this in the future
                ephemeris["Toc"] = toc

                # Compute the satellite coordinates at the emission epoch
                sv_pos.append(
                    self.ephemeris_processor(
                        t=emission_epoch[sv],
                        metadata=nav_metadata,
                        data=ephemeris,
                        system_time=(
                            True
                            if (approx_rcvr is not None and approx_sv is not None)
                            else False
                        ),
                    )
                )

            # NOTE: The "system time" kwargs is used here to compute the satellite coordinates at the emission epoch given
            # the emission epoch on GPS time. If the approximate receiver location and approximate satellite location are available,
            # then the system time is set to True. This is because the satellite clock bias correction should not be applied to the
            # to the emission epoch. However, if the approximate receiver location and approximate satellite location are not available,
            # the pseudorange is used to calculate the travel time. In this case, the satellite clock bias correction must be applied to the
            # emission epoch since the pseudorange contains the satellite clock bias correction.

            # Stack the satellite coordinates into a DataFrame
            sv_coords = pd.DataFrame(
                data=sv_pos, index=nav.index.get_level_values("sv")
            )

        # Compute the sagna effect to correct for the Earth's rotation
        sv_coords[["x", "y", "z"]] = sagnac_correction(
            sv_position=sv_coords[["x", "y", "z"]].to_numpy(dtype=np.float64),
            dt=travel_time,
        )

        return sv_coords

    def compute_sv_coordinates(
        self,
        reception_time: pd.Timestamp,
        navigation_data: pd.DataFrame,
        navigation_metadata: pd.Series,
        pseudorange: pd.Series,
        approx_receiver_location: pd.Series,
        is_sp3: bool = False,
    ) -> pd.DataFrame:
        """Iteratively compute the satellite coordinates at the reception epoch.

        This iteratively computes the satellite coordinates at the reception epoch by applying the satellite clock correction and the Sagnac effect.
        Consecutive iterations are performed to ensure the satellite coordinates are accurate.


        Args:
            reception_time (pd.Timestamp): Reception time of the GPS observations.
            navigation_data (pd.DataFrame): Navigation data.
            navigation_metadata (pd.Series): Metadata for the navigation data.
            pseudorange (pd.Series): Pseudorange of the GPS observations.
            approx_receiver_location (pd.Series, optional): Approximate receiver location in ECEF coordinate. Defaults to None.
            is_sp3 (bool, optional): If True, then the navigation data is in SP3 format. Defaults to False.

        Returns:
            pd.DataFrame: The processed satellite coordinates at the reception epoch.
        """
        # Get the satellite coordinates at the emission epoch
        sv_coords_at_i = self.compute_sv_coordinates_at_emission_epoch(
            reception_time=reception_time,
            nav=navigation_data,
            nav_metadata=navigation_metadata,
            pseudorange=pseudorange,
            approx_receiver_location=approx_receiver_location[["x", "y", "z", "cdt"]],
            approx_sv_location=None,
            is_sp3=is_sp3,
        )

        # Iterate over the satellite coordinates to correct for the satellite clock bias
        for _ in range(self.SV_COORD_CORRECTION_ITER):
            # Compute the satellite coordinates at the reception epoch
            sv_coords_at_i = self.compute_sv_coordinates_at_emission_epoch(
                reception_time=reception_time,
                nav=navigation_data,
                nav_metadata=navigation_metadata,
                pseudorange=pseudorange,
                approx_receiver_location=approx_receiver_location[
                    ["x", "y", "z", "cdt"]
                ],
                approx_sv_location=sv_coords_at_i[["x", "y", "z"]],
                is_sp3=is_sp3,
            )

        return sv_coords_at_i

    def bootstrap(
        self,
        epoch: Epoch,
    ) -> pd.Series:
        """Bootstrap the receiver location using the WLS.

        This is used to get an approximate receiver location for the WLS triangulation
        which is further used to compute the satellite coordinates at the emission epoch.

        Args:
            epoch (Epoch): Epoch containing observation data and navigation data.

        Returns:
            pd.Series: The approximate receiver location in ECEF coordinate.
        """
        # Process the satellite coordinates at the reception epoch
        coords = self.compute_sv_coordinates_at_emission_epoch(
            reception_time=epoch.timestamp,
            nav=epoch.nav_data,
            nav_metadata=epoch.nav_meta,
            pseudorange=epoch.obs_data[Epoch.L1_CODE_ON],
            approx_receiver_location=None,
            approx_sv_location=None,
            is_sp3=epoch.profile.get("navigation_format", "ephemeris") == "sp3",
        )
        # Correct for the satellite clock bias
        if epoch.profile.get("mode", "single") == "dual":
            pseudorange = ionosphere_free_combination(
                p1=epoch.obs_data[self.L1_CODE_ON].to_numpy(dtype=np.float64),
                p2=epoch.obs_data[self.L2_CODE_ON].to_numpy(dtype=np.float64),
            )
        else:
            pseudorange = epoch.obs_data[self.L1_CODE_ON].to_numpy(dtype=np.float64)

        # Correct for the satellite clock bias
        pseudorange = (
            pseudorange + SPEED_OF_LIGHT * coords[IGPSEphemeris.SV_CLOCK_BIAS_KEY]
        )

        # Get the base solution using WLS
        base_solution = wls_triangulation(
            pseudorange=pseudorange.to_numpy(dtype=np.float64),
            sv_pos=coords[["x", "y", "z"]].to_numpy(dtype=np.float64),
            x0=np.zeros(4),
            max_iter=1000,
            eps=1e-7,
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
            max_iter=100,
        )

        # Attach the latitude, longitude and height to the solution
        solution["lat"] = lat
        solution["lon"] = lon
        solution["height"] = height

        return solution

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

        #############################  Compute the Satellite Coordinates #############################
        # Get the satellite coordinates at the emission epoch
        coords = self.compute_sv_coordinates(
            reception_time=epoch.timestamp,
            navigation_data=epoch.nav_data,
            navigation_metadata=epoch.nav_meta,
            pseudorange=epoch.obs_data[self.L1_CODE_ON],
            approx_receiver_location=epoch.approximate_coords,
            is_sp3=epoch.profile.get("navigation_format", "ephemeris") == "sp3",
        )

        # Get the elevation and azimuth of the satellites and attach it to the coords
        coords["elevation"], coords["azimuth"] = elevation_and_azimuthal(
            satellite_positions=coords[["x", "y", "z"]].to_numpy(dtype=np.float64),
            observer_position=epoch.approximate_coords[["x", "y", "z"]].to_numpy(
                dtype=np.float64
            ),
        )

        #############################  Smoothing the Epoch Data #############################
        # If the epoch is smoothed, then apply swap the smoothed key as C1C
        if epoch.is_smoothed:
            if BaseSmoother.SMOOOTHING_KEY in epoch.obs_data.columns:
                epoch.obs_data["C1C"] = epoch.obs_data[
                    BaseSmoother.SMOOOTHING_KEY
                ]  # Swap the smoothed key as C1C

        #############################  Print the Epoch Information #############################
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

        ######################### Dual Frequency Processing #########################
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
            codeEF = ionosphere_free_combination(
                p1=epoch.obs_data[self.L1_CODE_ON].to_numpy(dtype=np.float64),
                p2=epoch.obs_data[self.L2_CODE_ON].to_numpy(dtype=np.float64),
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
            phaseEF = ionosphere_free_combination(
                p1=epoch.obs_data[self.L1_PHASE_ON].to_numpy(dtype=np.float64)
                * L1_WAVELENGTH,  # Scale the L1 phase to meters
                p2=epoch.obs_data[self.L2_PHASE_ON].to_numpy(dtype=np.float64)
                * L2_WAVELENGTH,  # Scale the L2 phase to meters
            )

        ######################### Single Frequency Processing #########################
        elif epoch.profile.get("mode", "single") == "single":
            # Get the pseudorange
            codeEF = epoch.obs_data[self.L1_CODE_ON].to_numpy(dtype=np.float64)
            phaseEF = (epoch.obs_data[self.L1_PHASE_ON] * L1_WAVELENGTH).to_numpy(
                dtype=np.float64
            )

        else:
            raise ValueError(
                f"Invalid mode flag: {epoch.profile.get('mode', 'single')}. Supported modes are ['dual', 'single']"
            )

        ### Below are the corrections that can be applied to the pseudorange and phase measurements ###

        ######################### Apply the Satellite Clock Correction #########################
        # Apply the satelllite clock correction to the pseudorange and phase measurements
        # The clock bias is calculated as the product of the satellite clock bias and the speed of light
        codeEF = (
            codeEF + SPEED_OF_LIGHT * coords[IGPSEphemeris.SV_CLOCK_BIAS_KEY].values
        )
        phaseEF = (
            phaseEF + SPEED_OF_LIGHT * coords[IGPSEphemeris.SV_CLOCK_BIAS_KEY].values
        )

        ######################### Apply the Tropospheric Correction #########################
        if epoch.profile.get("apply_tropo", False):
            coords["tropospheric_correction"] = self.compute_tropospheric_correction(
                day_of_year=epoch.timestamp.dayofyear,
                sv_coords=coords,
                approx_receiver_location=epoch.approximate_coords,
                model=epoch.profile.get("tropospheric_model", "saastamoinen"),
            )
            # Correct the pseudorange for the tropospheric correction
            codeEF = (
                codeEF - coords["tropospheric_correction"].values
            )  # Do not change this to -= as it will change the original data!
            phaseEF = (
                phaseEF - coords["tropospheric_correction"].values
            )  # Do not change this to -= as it will change the original data!

        ######################### Apply the Ionospheric Correction #########################
        # Only apply ionospheric correction for single mode processing
        if (
            epoch.profile.get("apply_iono", False)
            and epoch.profile.get("mode", "dual") == "single"
        ):

            # Check if the ionospheric correction parameters are available
            if epoch.nav_meta.get("IONOSPHERIC CORR", None) is None:
                raise ValueError(
                    "Invalid ionospheric correction parameters. Must contain ['IONOSPHERIC CORR'] in the navigation metadata."
                )

            coords["ionospheric_correction"] = self.compute_ionospheric_correction(
                sv_coords=coords,
                approx_receiver_location=epoch.approximate_coords,
                time=epoch.timestamp,
                iono_params=epoch.nav_meta.get("IONOSPHERIC CORR", None),
            )
            # Apply the ionospheric correction
            codeEF = codeEF - coords["ionospheric_correction"].values

            # Apply the ionospheric correction to the phase measurements
            # NOTE: The ionospheric correction sign is negative in the phase measurements
            phaseEF = phaseEF - coords["ionospheric_correction"].values

        ######################### Return the Pseudorange and Satellite States #########################
        return (
            pd.DataFrame(
                data=np.column_stack([codeEF, phaseEF]),
                index=epoch.obs_data.index,
                columns=[Epoch.L1_CODE_ON, Epoch.L1_PHASE_ON],
            ),
            coords,
        )

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
