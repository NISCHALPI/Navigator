"""Dual Frequency Unscented Kalman Method for Triangulation Interface.

Supported Constellations:
    - GPS

This module implements the Unscented Kalman Filter for GPS triangulation.

State Definition:
    The state vector (curr_state) represents the receiver and is defined as follows:
    
    x = [latitude, latitude_velocity, longitude, longitude_velocity, height, height_velocity, clock_drift, clock_drift_rate, wet_tropospheric_delay, B1, ..., Bn]

    Where:
    - latitude: Latitude of the receiver.
    - latitude_velocity: Velocity of the latitude.
    - longitude: Longitude of the receiver.
    - longitude_velocity: Velocity of the longitude.
    - height: Height of the receiver.
    - height_velocity: Velocity of the height.
    - clock_drift: Clock drift.
    - clock_drift_rate: Clock drift rate.
    - wet_tropospheric_delay: Wet tropospheric delay.
    - B: Bias of the phase measurements, including integer ambiguity and hardware delay.

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
from ......utility.transforms.coordinate_transforms import geocentric_to_ellipsoidal
from ..tools.phase_based.cartesian_state.measurement import phase_measurement_model
from ..tools.phase_based.cartesian_state.noise import phase_process_noise_profile
from ..tools.phase_based.cartesian_state.state_transistion import (
    phase_state_transistion_matrix,
)
from .ikalman_interface import IKalman

__all__ = ["PhaseUnscentedKalmanTriangulationInterface"]


class PhaseUnscentedKalmanTriangulationInterface(IKalman):
    """Unscented Kalman Method for Triangulation Interface.

    The state vector is parameterized in terms of the receiver's position, velocity,
    clock bias, and wet tropospheric delay.

    This class implements the Unscented Kalman Filter for triangulation.
    A constant velocity model is assumed for the state transition function.
    """

    state = [
        "x",
        "x_dot",
        "y",
        "y_dot",
        "z",
        "z_dot",
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
        S_x: float = 0.2,
        S_y: float = 0.2,
        S_z: float = 0.2,
        S_wet: float = 0.1,
        S_b: float = 100,
        S_code: float = 5,
        S_phase: float = 2,
        h_0: float = 2e-21,
        h_2: float = 2e-23,
        initial_guess: Series = None,
    ) -> None:
        """Initializes an instance of the Unscented Kalman Method for Triangulation.

        Args:
            num_sv (int): The number of satellites to track.
            dt (float): The sampling time interval in seconds.
            sigma_r (float): The standard deviation of the measurement noise.
            S_x (float): The white noise spectral density for the random walk position error in the x-direction.
            S_y (float): The white noise spectral density for the random walk position error in the y-direction.
            S_z (float): The white noise spectral density for the random walk position error in the z-direction.
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
        if S_x < 0 or S_y < 0 or S_z < 0 or S_wet < 0 or S_b < 0:
            raise ValueError(
                "The white noise spectral density for the random walk position error must be non-negative."
            )
        self.S_x, self.S_y, self.S_z = S_x, S_y, S_z
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
            S_x=self.S_x,
            S_y=self.S_y,
            S_z=self.S_z,
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

        # Initialize the state vector
        # Use coordinates of Washington DC as the initial guess for the state vector
        self.filter.x[[0, 2, 4]] = np.array(
            [1115077.69025948, -4843958.49112974, 3983260.99261736]
        )
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
        self.P = np.copy(self.filter.Q)

        # Initialize the initial sv_map
        self.sv_map = {}

    def fx(self, x: np.ndarray, dt: float) -> np.ndarray:  # noqa : ARG002
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
        lat, long, height = geocentric_to_ellipsoidal(
            x=state[0], y=state[2], z=state[4]
        )
        # Return the state vector as a pandas series
        return_val = Series(
            {
                "x": state[0],
                "y": state[2],
                "z": state[4],
                "x_dot": state[1],
                "y_dot": state[3],
                "z_dot": state[5],
                "lat": lat,
                "lon": long,
                "height": height,
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


# Path: src/navigator/core/triangulate/itriangulate/kalman/interfaces/phase_based_unscented_kalman_interface.py
