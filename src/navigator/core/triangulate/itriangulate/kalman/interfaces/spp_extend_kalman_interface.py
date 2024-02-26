"""This module contains the ITriangulate interface for the extended Kalman filter.

Classes:
    ExtendedKalmanInterface

State Definitions:
    state = [x,x_dot, y, y_dot,  z, z_dot, cdt, cdt_dot]

    where:
        x, y, z: position in the ECEF frame
        x_dot, y_dot, z_dot: velocity in the ECEF frame
        cdt: clock bias in meters
        cdt_dot: clock drift in meters per second

Measurement Definitions:
    measurements  =  [Pseudorange]

    where:
        Pseudorange: pseudorange measurement in meters

    Error Model:
        The tropospheric delay and ionospheric delay are preapplyed to the pseudorange measurement by means of models or range combination.

"""

import numpy as np
from pandas.core.api import DataFrame, Series

from ......epoch.epoch import Epoch
from ......filters.extended.ekf import ExtendedKalmanFilter
from ..tools.spp.default_noise_models import (
    measurement_noise_profile,
    octa_state_process_noise_profile,
)
from ..tools.spp.measurement_model import (
    jacobian_measurement_function,
    measurement_function,
)
from ..tools.spp.state_transistion import constant_velocity_state_transistion
from .ikalman_interface import IKalman


class ExtendedKalmanInterface(IKalman):
    """This class provides the interface for the extended Kalman filter.

    The extended Kalman filter is used to estimate the state of the system given a series of measurements.


    Reference:
        https://apps.dtic.mil/sti/tr/pdf/AD1010622.pdf

    """

    state = ["x", "x_dot", "y", "y_dot", "z", "z_dot", "cdt", "cdt_dot"]
    measurements = ["Pseudorange"]

    def __init__(
        self,
        num_sv: int,
        dt: float,
        sigma_r: float = 6,
        S_x: float = 0,
        S_y: float = 0,
        S_z: float = 0,
        h_0: float = 2e-21,
        h_2: float = 2e-23,
        initial_guess: Series = None,
    ) -> None:
        """Initialize the ExtendedKalmanInterface class.

        Args:
            num_sv (int): The number of satellites to track  using the extended Kalman filter.
            sigma_r: The standard deviation of the pseudorange measurement noise.
            S_x (float): The white noise spectral density for the random walk position error in the x-direction.
            S_y (float): The white noise spectral density for the random walk position error in the y-direction.
            S_z (float): The white noise spectral density for the random walk position error in the z-direction.
            dt (float): The sampling time interval in seconds.
            h_0 (float, optional): The coefficients of the power spectral density of the clock noise. Defaults to 2e-21.
            h_2 (float, optional): The coefficients of the power spectral density of the clock noise. Defaults to 2e-23.
            initial_guess (Series, optional): The initial guess for the state vector. Defaults to None.

        Returns:
            None
        """
        super().__init__(num_sv=num_sv, dt=dt, interface_id="ExtendedKalmanFilter")

        # The standard deviation of the pseudorange measurement noise
        if sigma_r <= 0:
            raise ValueError(
                "The standard deviation of the pseudorange measurement noise must be positive."
            )

        self.sigma_r = sigma_r

        # The white noise spectral density for the random walk position error in the x-direction
        if S_x < 0 or S_y < 0 or S_z < 0:
            raise ValueError(
                "The white noise spectral density for the random walk position error must be non-negative."
            )

        self.S_x, self.S_y, self.S_z = S_x, S_y, S_z

        # The coefficients of the power spectral density of the clock noise
        if h_0 <= 0 or h_2 <= 0:
            raise ValueError(
                "The coefficients of the power spectral density of the clock noise must be positive."
            )

        self.h_0, self.h_2 = h_0, h_2

        # Initialize the extended Kalman filter
        self.filter = ExtendedKalmanFilter(
            dim_x=len(self.state),
            dim_y=self.num_sv,
            innovation_window=30,
            adjust_after=500,
        )
        # Add the process noise profile
        self.filter.Q = octa_state_process_noise_profile(
            S_x=self.S_x,
            S_y=self.S_y,
            S_z=self.S_z,
            h_0=self.h_0,
            h_2=self.h_2,
            dt=self.dt,
        )
        # Add the measurement noise profile
        self.filter.R = measurement_noise_profile(self.sigma_r, self.num_sv)

        # Set the initial guess for the state vector
        if initial_guess is not None:
            self.filter.x[6] = (
                initial_guess["cdt"] if "cdt" in initial_guess else 1e-5 * 299792458
            )
            self.filter.x[[0, 2, 4]] = initial_guess[["x", "y", "z"]].values

        # Set the initial guess for the state covariance matrix
        self.filter.P *= 1000
        # Set the Initial Process Noise of Clock Estimate
        self.filter.P[6, 6] = 2 * self.filter.Q[6, 6]

    def process_state(self, state: np.ndarray) -> Series:
        """Process the state vector.

        Args:
            state (np.ndarray): The state vector.

        Returns:
            Series: A pandas series containing the output provided to the user.
        """
        lat, lon, height = self._clipped_geocentric_to_ellipsoidal(
            state[0], state[2], state[4]
        )

        return Series(
            {
                "x": state[0],
                "y": state[2],
                "z": state[4],
                "lat": lat,
                "lon": lon,
                "height": height,
                "x_dot": state[1],
                "y_dot": state[3],
                "z_dot": state[5],
                "cdt": state[6],
                "cdt_dot": state[7],
            }
        )

    def predict_update_loop(
        self, ranges: np.ndarray, sv_coords: np.ndarray
    ) -> np.ndarray:
        """Runs the predict and update loop of the Kalman filter.

        Args:
            ranges (np.ndarray): The range measurements.
            sv_coords (np.ndarray): The coordinates of the satellites.

        Returns:
            np.ndarray: The state vector after the predict and update loop.
        """
        # Predict the state and covariance matrix
        # Predict the state and covariance matrix
        # Note: Data Structures passed to the filter are numpy arrays not pandas series or dataframes

        residual = self.filter.predict_update(
            y=ranges,
            F=constant_velocity_state_transistion(
                x=np.eye(len(self.state)), dt=self.dt
            ),  # This is a trick to get the state transition matrix  instead of the state transition function
            hx=measurement_function,
            HJacobian=jacobian_measurement_function,
            hx_kwargs={"sv_location": sv_coords},
            HJ_kwargs={"sv_location": sv_coords},
            u=None,  # No control input for GPS problems
        )
        return self.filter._x_post, residual

    def epoch_profile(self) -> str:
        """Get the epoch profile for the respective Kalman filter.

        Some might apply Ionospheric correction, tropospheric correction, etc while
        others might not. This is controlled by the epoch profile set to the epoch.

        Returns:
            str: The epoch profile that is updated for each epoch.
        """
        return {"apply_tropo": True, "mode": "dual", "apply_iono": True}

    def _compute(
        self,
        epoch: Epoch,
        *args,  # noqa : ARG002
        **kwargs,
    ) -> Series | DataFrame:
        """Computes the triangulated position using the Unscented Kalman Filter.

        Args:
            epoch (Epoch):  The epoch to be processed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Series | DataFrame: A pandas series containing the triangulated position, DOPS, and clock bias and drift.
        """
        # Add the needed epoch profile to the epoch before processing
        epoch.profile = self.epoch_profile()

        # Get the range and sv_coordinates
        pseudorange, sv_coordinates = self._preprocess(
            epoch=epoch,
            **kwargs,
        )

        # Remove the extra satellites from the epoch
        # Use the elevation trim to remove the extra satellites
        pseudorange, sv_coordinates = self._trim_by_elevation(
            pseudorange=pseudorange,
            sv_coords=sv_coordinates,
            observer_position=self.filter.x[[0, 2, 4]],
        )

        # Run the predict and update loop
        # Note : pass the numpy arrays to the filter instead of the pandas series or dataframes
        rawstate, residual = self.predict_update_loop(
            ranges=pseudorange.values,
            sv_coords=sv_coordinates[["x", "y", "z"]].values,
        )

        # Process the state
        state = self.process_state(state=rawstate)

        # Add the residual to processed state
        state["residual"] = residual

        return state
