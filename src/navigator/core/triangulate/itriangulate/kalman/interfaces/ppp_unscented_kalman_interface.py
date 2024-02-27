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
from ..tools.code_based.default_noise_models import octa_state_process_noise_profile
from ..tools.phase_based.measurement import ppp_measurement_model
from ..tools.phase_based.noise import ppp_process_noise
from ..tools.phase_based.state_transistion import ppp_state_transistion_matrix
from .ikalman_interface import IKalman

__all__ = ["PhaseBasedUnscentedKalmanTriangulationInterface"]


class PhaseBasedUnscentedKalmanTriangulationInterface(IKalman):
    """Unscented Kalman Method for Triangulation Interface.

    This class implements the Unscented Kalman Filter for triangulation.
    A constant velocity model is assumed for the state transition function.
    """

    L1_CODE_ON = "C1C"
    L2_CODE_ON = "C2W"
    L1_PHASE_ON = "L1C"
    L2_PHASE_ON = "L2W"

    L1_WAVELENGTH = 0.1902936727983649
    L2_WAVELENGTH = 0.2442102134245683

    state = [
        "x",  # x position
        "x_dot",  # x velocity
        "y",  # y position
        "y_dot",  # y velocity
        "z",  # z position
        "z_dot",  # z velocity
        "cdt",  # clock bias
        "cdt_dot",  # clock drift
        "t_wet",  # wet tropospheric delay
        "lambda_N",  # integer ambiguity of num_sv satellites i.e num_sv integer ambiguity
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

        # Set the state transition function
        self.F = ppp_state_transistion_matrix(dt=self.dt, num_sv=self.num_sv)

        # Initialize the Unscented Kalman Filter
        self.filter = UnscentedKalmanFilter(
            dim_x=(9 + 1 * self.num_sv),  # Number of state variables
            dim_z=self.num_sv
            * 2,  # Number of measurement variables count twice since we have code and phase measurements
            dt=dt,  # Sampling time interval
            hx=ppp_measurement_model,  # Measurement function
            fx=self.fx,  # State transition function
            points=MerweScaledSigmaPoints(
                n=(9 + 1 * self.num_sv), alpha=0.1, beta=2.0, kappa=-1
            ),
        )

        # Initialize the Unscented Kalman Filter
        self.filter.Q = ppp_process_noise(
            sigma_trop=sigma_tropr,
            sigma_bias=sigma_bias,
            num_sv=self.num_sv,
        )
        # self.filter.Q[:8, :8] = octa_state_process_noise_profile(
        #     S_x=self.S_x,
        #     S_y=self.S_y,
        #     S_z=self.S_z,
        #     dt=self.dt,
        # )

        # Set the measurement noise covariance matrix
        self.filter.R = np.eye(self.num_sv * 2) * self.sigma_r**2
        # Set the phase measurement noise covariance matrix
        self.filter.R[self.num_sv :, self.num_sv :] /= 1000

        # Set the initial guess for the state vector
        if initial_guess is not None:
            self.filter.x[6] = (
                initial_guess["cdt"] if "cdt" in initial_guess else 1e-5 * 299792458
            )
            self.filter.x[[0, 2, 4]] = initial_guess[["x", "y", "z"]].values

        # # Set the initial guess for the state covariance matrix
        # self.filter.P

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
        lat, lon, height = self._clipped_geocentric_to_ellipsoidal(
            state[0], state[2], state[4]
        )
        # Return the state vector as a pandas series
        return_val = Series(
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
            }
        )

        # Update the integer ambiguity, code and phase bias
        for i in range(self.num_sv):
            return_val[f"lambda_N{i}"] = state[9 + i]

        return return_val

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
        self.filter.update(z=psedurange, sv_matrix=sv_coords, num_sv=self.num_sv)
        return self.filter.x_post

    def epoch_profile(self) -> str:
        """Get the epoch profile for the respective Kalman filter.

        Some might apply Ionospheric correction, tropospheric correction, etc while
        others might not. This is controlled by the epoch profile set to the epoch.

        Returns:
            str: The epoch profile that is updated for each epoch.
        """
        return {"apply_tropo": False, "mode": "single", "apply_iono": False}

    def _get_pseudorange(self, epoch: Epoch) -> np.ndarray:
        """Get the IF free pseudorange measurements for the current epoch.

        Args:
            epoch (Epoch): The epoch to be processed.

        Returns:
            np.ndarray: The pseudorange measurements for the current epoch.
        """
        # Get the ionosphere free combination
        l1, l2 = epoch.obs_data[self.L1_PHASE_ON], epoch.obs_data[self.L2_PHASE_ON]
        c1, c2 = epoch.obs_data[self.L1_CODE_ON], epoch.obs_data[self.L2_CODE_ON]

        # Scale the phase measurements to meters
        l1 = l1 * self.L1_WAVELENGTH
        l2 = l2 * self.L2_WAVELENGTH

        # Get the ionosphere free combination
        ifl = ionosphere_free_combination(l1.to_numpy(), l2.to_numpy())
        ifc = ionosphere_free_combination(c1.to_numpy(), c2.to_numpy())

        # Stack and return the measurements
        return np.hstack((ifl, ifc))

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

        # Get the pseudorange measurements
        pseudorange = self._get_pseudorange(epoch)

        # Get the satellite coordinates
        epoch.profile = epoch.INITIAL
        _, sv_coords = self._preprocess(epoch)

        # Run the predict and update loop
        state = self.predict_update_loop(
            pseudorange, sv_coords[["x", "y", "z"]].to_numpy()
        )

        # Process the state vector
        return self.process_state(state)
