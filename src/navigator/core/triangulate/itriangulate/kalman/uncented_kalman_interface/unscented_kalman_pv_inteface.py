"""Unscented Kalman Method for Triangulation.

This module implements the Unscented Kalman Method (UKM) for triangulation, specifically designed for the pseudorange GPS problem. UKM is a variant of the Kalman Filter that employs the Unscented Transform to estimate the next state vector. Given the non-linearity in the measurement function of the GPS problem, UKM is preferred over the traditional Kalman Filter. The dynamic model assumed here is a constant velocity model.

The state vector is defined as follows:
x = [x, x_dot, y, y_dot, z, z_dot, t, t_dot]

Functions:
- `fx`: State transition function that converts the state vector into the next state vector.
- `hx`: Measurement function for the pseudorange GPS problem, converting the state vector into a pseudorange measurement vector.

Filter Backend:
    filterpy.kalman.UnscentedKalmanFilter

See Example At:
    src/docs/intro/unscenetd_kalman_filter_gps.ipynb
"""

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from pandas.core.api import DataFrame, Series

from ......epoch.epoch import Epoch
from ..kalman_interface import KalmanTriangulationInterface
from ..tools.default_noise_models import (
    measurement_noise_profile,
    octa_state_process_noise_profile,
)
from ..tools.measurement_model import measurement_function as hx
from ..tools.state_transistion import constant_velocity_state_transistion as fx

__all__ = ['UnscentedKalmanTriangulationInterface']


class UnscentedKalmanTriangulationInterface(KalmanTriangulationInterface):
    """Unscented Kalman Method for Triangulation Interface.

    This class implements the Unscented Kalman Filter for triangulation.
    A constant velocity model is assumed for the state transition function.
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
        """Initializes an instance of the Unscented Kalman Method for Triangulation.

        Args:
            num_sv (int): The number of satellites to track.
            dt (float): The sampling time interval in seconds.
            sigma_r (float): The standard deviation of the measurement noise.
            S_x (float): The standard deviation of the process noise in the x-direction.
            S_y (float): The standard deviation of the process noise in the y-direction.
            S_z (float): The standard deviation of the process noise in the z-direction.
            h_0 (float): The constant term in the clock bias and drift model.
            h_2 (float): The quadratic term in the clock bias and drift model.
            initial_guess (Series): The initial guess for the state vector.

        Raises:
            ValueError: If the number of satellites to track is less than 2.
            ValueError: If the sampling time interval is less than or equal to 0.
        """
        super().__init__(num_sv=num_sv, dt=dt, interface_id="UncentedKalmanInterface")

        # Set the measurement noise standard deviation
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

        # Initialize the Unscented Kalman Filter
        self.filter = UnscentedKalmanFilter(
            dim_x=len(self.state),  # Number of state variables
            dim_z=self.num_sv,  # Number of measurement variables
            dt=dt,  # Sampling time interval
            fx=fx,  # State transition function
            hx=hx,  # Measurement function
            points=MerweScaledSigmaPoints(
                n=len(self.state), alpha=0.1, beta=2.0, kappa=-1
            ),
        )

        # Initialize the Unscented Kalman Filter
        self.filter.Q = octa_state_process_noise_profile(
            S_x=self.S_x,
            S_y=self.S_y,
            S_z=self.S_z,
            h_0=self.h_0,
            h_2=self.h_2,
            dt=self.dt,
        )

        # Set the measurement noise covariance matrix
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
            Series: A pandas series containing the state vector.
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
        self.filter.predict()
        self.filter.update(
            z=ranges,  # The range measurements
            sv_location=sv_coords,  # The coordinates of the satellites
        )

        return self.filter.x_post

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

        return self.process_state(
            self.predict_update_loop(
                ranges=pseudorange.values,
                sv_coords=sv_coordinates[["x", "y", "z"]].values,
            )
        )
