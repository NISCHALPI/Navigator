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
        Pseudorange: pseudorange measurement in meters.
"""

from pandas.core.api import DataFrame as DataFrame, Series

from navigator.epoch.epoch import Epoch
from .ikalman import IKalman
from ..algos.dynamics_model.constant_velocity import Q, G, h, HJacobian
from filterpy.kalman import ExtendedKalmanFilter
import numpy as np

__all__ = ["ExtendedKalmanInterface"]


class ExtendedKalmanInterface(IKalman):
    """This class provides the interface for the extended Kalman filter for GPS/GNSS triangulation.

    References:
        -  https://apps.dtic.mil/sti/tr/pdf/AD1010622.pdf
    """

    state = ["x", "x_dot", "y", "y_dot", "z", "z_dot", "cdt", "cdt_dot"]
    measurements = ["pseudorange"]

    def __init__(
        self,
        dt: float,
        num_sv: int,
        x_initial: np.ndarray,
        P_initial: np.ndarray,
        R: np.ndarray,
        Q_0: np.ndarray,
        log_innovation: bool = False,
    ) -> None:
        """Initializes the ExtendedKalmanInterface object.

        Args:
            dt (float): Time step.
            num_sv (int): Number of satellites.
            x_initial (np.ndarray): Initial state vector estimate.
            P_initial (np.ndarray): Initial error covariance matrix estimate.
            R (np.ndarray): Measurement noise covariance matrix.
            Q_0 (np.ndarray): Power spectral density of the continuous process white noise.
            log_innovation (bool): Flag to log the innovation.
        """
        super().__init__(
            num_sv=num_sv,
            dt=dt,
            filter_name="Extended Kalman Filter",
        )

        # Initialize the filter
        self._filter = ExtendedKalmanFilter(
            dim_x=8,
            dim_z=num_sv,
            dim_u=0,
        )

        # Initialize the estimates
        self._filter.x = x_initial
        self._filter.P = P_initial
        self._filter.R = R
        self._filter.Q = Q(dt=dt, autocorrelation=Q_0)

        # Initialize the dynamics model
        self.F = G(dt)

        # Initialize the log innovation flag
        self.log_innovation = log_innovation

    def process_state(self, state: np.ndarray) -> Series:
        """Processes the state vector into a pandas Series.

        Args:
            state (np.ndarray): State vector.

        Returns:
            Series: State vector as a pandas Series.
        """
        return Series(
            {
                "x": state[0],
                "x_dot": state[1],
                "y": state[2],
                "y_dot": state[3],
                "z": state[4],
                "z_dot": state[5],
                "cdt": state[6],
                "cdt_dot": state[7],
            }
        )

    def epoch_profile(self) -> str:
        """Epoch profile to be used for preprocessing the epoch."""
        return {"apply_tropo": False, "apply_iono": True, "mode": "dual"}

    def predict_update_loop(
        self, ranges: np.ndarray, sv_coords: np.ndarray
    ) -> np.ndarray:
        """Predicts and updates the state vector.

        Args:
            ranges (np.ndarray): Pseudorange measurements.
            sv_coords (np.ndarray): Satellite coordinates.

        Returns:
            np.ndarray: Updated state vector.
        """
        # Predict the state
        self._filter.predict_update(
            z=ranges, HJacobian=HJacobian, Hx=h, args=(sv_coords,), hx_args=(sv_coords,)
        )
        # Log the innovation
        # Return the updated state
        return self._filter.x

    def _compute(self, epoch: Epoch, *args, **kwargs) -> Series | DataFrame:
        """Computes the state vector estimate for the given epoch.

        Args:
            epoch (Epoch): Epoch object.

        Returns:
            Series | DataFrame: State vector estimate.
        """
        # Add the respective epoch profile
        if epoch.profile["mode"] != "dummy":
            epoch.profile = self.epoch_profile()

        # Preprocess the data
        ranges, sv_coords = self._preprocess(epoch=epoch, **kwargs)

        # Run the predict-update loop
        state = self.predict_update_loop(
            ranges=ranges.to_numpy(np.float64), sv_coords=sv_coords.to_numpy(np.float64)
        )

        # Process the state
        state = self.process_state(state)

        # Log the innovation if required
        if self.log_innovation:
            state["innovation"] = self._filter.y

        return state

    def _dim_state(self) -> int:
        return 8
