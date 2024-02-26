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
from ...algos.combinations import ionosphere_free_combination
from ..tools.ppp.measurement import ppp_measurement_model as hx
from ..tools.ppp.noise import ppp_process_noise
from ..tools.ppp.state_transistion import ppp_state_transistion_matrix as fx
from ..tools.spp.default_noise_models import octa_state_process_noise_profile
from .ikalman_interface import IKalman

__all__ = ["PPPUnscentedKalmanTriangulationInterface"]


class PPPUnscentedKalmanTriangulationInterface(IKalman):
    """Unscented Kalman Method for Triangulation Interface.

    This class implements the Unscented Kalman Filter for triangulation.
    A constant velocity model is assumed for the state transition function.
    """

    L1_CODE_ON = "C1C"
    L2_CODE_ON = "C2W"
    L1_PHASE_ON = "L1C"
    L2_PHASE_ON = "L2W"

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
        "lambda_N",
        "delta_code",
        "delta_phase",
    ]
    measurement = ["pseudorange", "carrier_phase"]

    def __init__(
        self,
        num_sv: int,
        dt: float,
        sigma_r: float = 6,
        S_x: float = 0.8,
        S_y: float = 0.8,
        S_z: float = 0.8,
        h_0: float = 2e-21,
        h_2: float = 2e-23,
        sigma_tropr: float = 10,
        sigma_bias: float = 10,
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
            sigma_tropr (float): The standard deviation of the tropospheric delay.
            sigma_bias (float): The standard deviation of the code and phase biases.
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
            dim_z=self.num_sv
            * 2,  # Number of measurement variables count twice since we have code and phase measurements
            dt=dt,  # Sampling time interval
            fx=fx,  # State transition function
            hx=hx,  # Measurement function
            points=MerweScaledSigmaPoints(
                n=len(self.state), alpha=0.1, beta=2.0, kappa=-1
            ),
        )

        # Initialize the Unscented Kalman Filter
        self.filter.Q = ppp_process_noise(
            sigma_trop=sigma_tropr,
            sigma_bias=sigma_bias,
        )
        self.filter.Q[:8, :8] = octa_state_process_noise_profile(
            S_x=self.S_x,
            S_y=self.S_y,
            S_z=self.S_z,
            dt=self.dt,
        )

        # Set the measurement noise covariance matrix
        self.filter.R = np.eye(self.num_sv * 2) * sigma_r**2

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
                "t_wet": state[8],
                "lambda_N": state[9],
                "delta_code": state[10],
                "delta_phase": state[11],
            }
        )

    def predict_update_loop(
        self, psedurange: np.ndarray, sv_coords: np.ndarray
    ) -> np.ndarray:
        """Runs the predict and update loop of the Kalman filter.

        Args:
            psedurange (np.ndarray): The pseudorange measurements.
            phase (np.ndarray): The carrier phase measurements.
            sv_coords (np.ndarray): The coordinates of the satellites.

        Returns:
            np.ndarray: The state vector after the predict and update loop.
        """
        self.filter.predict()
        self.filter.update(z=psedurange, sv_matrix=sv_coords)
        return self.filter.x_post

    def epoch_profile(self) -> str:
        """Get the epoch profile for the respective Kalman filter.

        Some might apply Ionospheric correction, tropospheric correction, etc while
        others might not. This is controlled by the epoch profile set to the epoch.

        Returns:
            str: The epoch profile that is updated for each epoch.
        """
        return {"apply_tropo": False, "mode": "single", "apply_iono": False}

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

        # Get the first n_sv satellites
        sv_coordinates = sv_coordinates.head(self.num_sv)
        # Get the crossponding observations measurements
        obs_data = epoch.obs_data.loc[sv_coordinates.index]

        # Get the phase and rangle measurements
        c1c = obs_data[self.L1_CODE_ON]
        c2w = obs_data[self.L2_CODE_ON]
        l1c = obs_data[self.L1_PHASE_ON] * (
            299792458 / 1575.42e6
        )  # L1 frequency to convert to meters
        l2w = obs_data[self.L2_PHASE_ON] * (
            299792458 / 1227.60e6
        )  # L2 frequency to convert to meters

        # Make a phase and code ion free combination
        il1 = ionosphere_free_combination(
            np.array(l1c, dtype=np.float64), np.array(l2w, dtype=np.float64)
        )
        ic1 = ionosphere_free_combination(
            np.array(c1c, dtype=np.float64), np.array(c2w, dtype=np.float64)
        )
        # Concanaate the measurements
        pseudorange = np.concatenate([ic1, il1], axis=0)

        return self.process_state(
            self.predict_update_loop(
                psedurange=pseudorange,
                sv_coords=sv_coordinates[["x", "y", "z"]].values,
            )
        )
