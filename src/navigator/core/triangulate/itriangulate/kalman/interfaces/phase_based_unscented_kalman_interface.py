"""Dual Frequency Unscented Kalman Method for Triangulation Interface.

Supported Constellations:
    - GPS

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
from ......utility.transforms.coordinate_transforms import ellipsoidal_to_geocentric
from ..tools.phase_based.measurement import phase_measurement_model
from ..tools.phase_based.noise import phase_process_noise_profile
from ..tools.phase_based.state_transistion import phase_state_transistion_matrix
from .ikalman_interface import IKalman

__all__ = ["PhaseBasedUnscentedKalmanTriangulationInterface"]


class PhaseBasedUnscentedKalmanTriangulationInterface(IKalman):
    """Unscented Kalman Method for Triangulation Interface.

    This class implements the Unscented Kalman Filter for triangulation.
    A constant velocity model is assumed for the state transition function.
    """

    state = [
        "lat",
        "lat_dot",
        "lon",
        "lon_dot",
        "height",
        "height_dot",
        "cdt",
        "cdt_dot",
        "t_wet",
        "Bias",  # bias term containing the integer ambiguity, as well as the code and phase biases
    ]
    measurement = ["pseudorange", "carrier_phase"]

    def __init__(
        self,
        num_sv: int,
        dt: float,
        S_lat: float = 1e-5,
        S_lon: float = 1e-5,
        S_h: float = 10,
        S_wet: float = 0.1,
        S_b: float = 100,
        S_code: float = 5,
        S_phase: float = 1,
        h_0: float = 2e-21,
        h_2: float = 2e-23,
        initial_guess: Series = None,
    ) -> None:
        """Initializes an instance of the Unscented Kalman Method for Triangulation.

        Args:
            num_sv (int): The number of satellites to track.
            dt (float): The sampling time interval in seconds.
            sigma_r (float): The standard deviation of the measurement noise.
            S_lat (float): The white noise spectral density for the random walk position error in the latitude direction.
            S_lon (float): The white noise spectral density for the random walk position error in the longitude direction.
            S_h (float): The white noise spectral density for the random walk position error in the height direction.
            h_0 (float): The constant term in the clock bias and drift model.
            h_2 (float): The quadratic term in the clock bias and drift model.
            S_wet (float): The white noise spectral density for the random walk position error in the wet tropospheric delay.
            S_b (float): The white noise spectral density for the random walk position error in the phase bias.
            S_code (float): The standard deviation of the pseudorange measurement noise.
            S_phase (float): The standard deviation of the phase measurement noise.
            initial_guess (Series): The initial guess for the state vector.

        Raises:
            ValueError: If the number of satellites to track is less than 2.
            ValueError: If the sampling time interval is less than or equal to 0.
        """
        super().__init__(num_sv=num_sv, dt=dt, interface_id="UncentedKalmanInterface")

        # Set the measurement noise standard deviation
        # The standard deviation of the pseudorange measurement noise
        if S_code < 0 or S_phase < 0:
            raise ValueError(
                "The standard deviation of the pseudorange measurement or phase noise must be non-negative."
            )

        self.S_code, self.S_phase = S_code, S_phase

        # The white noise spectral density for the random walk position error in the x-direction
        if S_lat < 0 or S_lon < 0 or S_h < 0 or S_wet < 0 or S_b < 0:
            raise ValueError(
                "The white noise spectral density for the random walk position error must be non-negative."
            )
        self.S_lat, self.S_lon, self.S_h = S_lat, S_lon, S_h
        self.S_wet, self.S_b = S_wet, S_b

        # The coefficients of the power spectral density of the clock noise
        if h_0 <= 0 or h_2 <= 0:
            raise ValueError(
                "The coefficients of the power spectral density of the clock noise must be positive."
            )

        self.h_0, self.h_2 = h_0, h_2

        # Set the state transition function
        self.F = phase_state_transistion_matrix(dt=self.dt, num_sv=self.num_sv)

        # Initialize the Unscented Kalman Filter
        self.filter = UnscentedKalmanFilter(
            dim_x=(9 + 1 * self.num_sv),  # Number of state variables
            dim_z=self.num_sv
            * 2,  # Number of measurement variables count twice since we have code and phase measurements
            dt=dt,  # Sampling time interval
            hx=phase_measurement_model,  # Measurement function
            fx=self.fx,  # State transition function
            points=MerweScaledSigmaPoints(
                n=(9 + 1 * self.num_sv), alpha=0.1, beta=2.0, kappa=-1
            ),
        )

        # Initialize the Unscented Kalman Filter
        self.filter.Q = phase_process_noise_profile(
            dt=self.dt,
            num_sv=self.num_sv,
            S_lat=self.S_lat,
            S_lon=self.S_lon,
            S_h=self.S_h,
            S_wet=self.S_wet,
            S_b=self.S_b,
            h_0=self.h_0,
            h_2=self.h_2,
        )

        # Set the measurement noise covariance matrix
        self.filter.R = np.eye(self.num_sv * 2) * self.S_code**2
        self.filter.R[self.num_sv :, self.num_sv :] = (
            np.eye(self.num_sv) * self.S_phase**2
        )

        # Set the phase measurement noise covariance matrix
        self.filter.R[self.num_sv :, self.num_sv :] /= 1000

        # Initialize the state vector
        # Use lat long of washington DC as the initial guess
        self.filter.x[[0, 2, 4]] = np.array([38.89511, -77.03637, 0])
        self.filter.x[[1, 3, 5]] = np.array([0, 0, 0])  # No initial velocity
        self.filter.x[6] = 1e-5 * 299792458  # Initial guess for the clock bias
        self.filter.x[7] = 0  # Initial guess for the clock drift

        # Set the initial guess for the state vector
        if initial_guess is not None:
            self.filter.x[6] = (
                initial_guess["cdt"] if "cdt" in initial_guess else 1e-5 * 299792458
            )
            self.filter.x[[0, 2, 4]] = initial_guess[["lat", "lon", "height"]].values

        # Set the initial guess for the state covariance matrix
        self.filter.P = np.eye(9 + 1 * self.num_sv) * 10
        # Set the initial guess for the state covariance matrix
        self.filter.P[[0, 2, 4], [0, 2, 4]] = np.array([S_lat, S_lon, S_h])
        self.filter.P[[1, 3, 5], [1, 3, 5]] = np.array([S_lat, S_lon, S_h]) * 0.1
        self.filter.P[6, 6] = 1e-5 * 299792458

        # Initialize the initial sv_map
        self.sv_map = {}

    def fx(self, x: np.ndarray, dt: float) -> np.ndarray:
        """State transition function that converts the state vector into the next state vector.

        Args:
            x (np.ndarray): The state vector.
            dt (float): The sampling time interval in seconds.

        Returns:
            np.ndarray: The next state vector.
        """
        return self.F @ x

    def process_state(self, state: np.ndarray) -> Series:
        """Process the state vector.

        Args:
            state (np.ndarray): The state vector.

        Returns:
            Series: A pandas series containing the state vector.
        """
        x, y, z = ellipsoidal_to_geocentric(lat=state[0], lon=state[2], height=state[4])
        # Return the state vector as a pandas series
        return_val = Series(
            {
                "x": x,
                "y": y,
                "z": z,
                "lat": state[0],
                "lon": state[2],
                "height": state[4],
                "lat_dot": state[1],
                "lon_dot": state[3],
                "height_dot": state[5],
                "cdt": state[6],
                "cdt_dot": state[7],
                "t_wet": state[8],
            }
        )

        # Update the integer ambiguity, code and phase bias
        for i in range(self.num_sv):
            return_val[f"B{i}"] = state[9 + i]

        return return_val

    def predict_update_loop(
        self, psedurange: Series, sv_coords: DataFrame, day_of_year: int
    ) -> np.ndarray:
        """Runs the predict and update loop of the Kalman filter.

        Args:
            psedurange (np.ndarray): The pseudorange measurements.
            phase (np.ndarray): The carrier phase measurements.
            sv_coords (np.ndarray): The coordinates of the satellites.
            day_of_year (int): The day of the year.

        Returns:
            np.ndarray: The state vector after the predict and update loop.
        """
        self.filter.predict()
        self.filter.update(
            z=psedurange.to_numpy(),
            sv_frame=sv_coords,
            num_sv=self.num_sv,
            day_of_year=day_of_year,
        )
        return self.filter.x_post

    def epoch_profile(self) -> str:
        """Get the epoch profile for the respective Kalman filter.

        Some might apply Ionospheric correction, tropospheric correction, etc while
        others might not. This is controlled by the epoch profile set to the epoch.

        Returns:
            str: The epoch profile that is updated for each epoch.
        """
        return {"apply_tropo": False, "mode": "phase", "apply_iono": False}

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
        # Initialize the sv map
        if len(self.sv_map) == 0:
            self.sv_map = epoch.common_sv

        # If the satellite is not in the sv_map, raise error since this can only be used
        # for continuous tracking of the same satellites
        if not all([sat in self.sv_map for sat in epoch.common_sv]):
            raise ValueError(
                "The satellites in the current epoch must be the same as the previous epoch since continuous tracking is assumed."
            )

        # Get the satellite coordinates
        epoch.profile = self.epoch_profile()
        pseudorange, sv_coords = self._preprocess(epoch, *args, **kwargs)

        # Run the predict and update loop
        state = self.predict_update_loop(
            pseudorange,
            sv_coords[["x", "y", "z", "elevation"]],
            day_of_year=epoch.timestamp.day_of_year,
        )

        # Process the state vector
        return self.process_state(state)
