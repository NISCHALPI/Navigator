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

import numpy as np
from pandas.core.api import DataFrame as DataFrame
from pandas.core.api import Series

from navigator.epoch.epoch import Epoch

from ..algos.dynamics_model.constant_velocity import G, HJacobian, Q, hx
from ..algos.kalman_filters.ekf import ExtendedKalmanFilter
from .ikalman import IKalman

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
        x0: np.ndarray,
        P0: np.ndarray,
        R: np.ndarray,
        Q_0: np.ndarray,
        log_innovation: bool = False,
        code_only: bool = False,
    ) -> None:
        """Initializes the ExtendedKalmanInterface object.

        Args:
            dt (float): Time step.
            num_sv (int): Number of satellites.
            x0 (np.ndarray): Initial state vector. (8,)
            P0 (np.ndarray): Initial state covariance matrix. (8, 8)
            R (np.ndarray): Measurement noise covariance matrix. (num_sv * 2, num_sv * 2)
            Q_0 (np.ndarray): Power spectral density of the continuous process white noise. (8, 8)
            log_innovation (bool): Flag to log the innovation.
            code_only (bool): Flag to indicate if the filter is used for code measurements or both code and carrier measurements.
        """
        super().__init__(
            num_sv=num_sv,
            dt=dt,
            filter_name="Extended Kalman Filter",
        )

        # Initialize the filter
        self._filter = ExtendedKalmanFilter(
            dim_x=8,
            dim_z=num_sv if code_only else 2 * num_sv,
        )

        # Initialize the state vector
        self.x0 = x0
        self.P0 = P0

        # Initialize the estimates
        self._filter.R = R
        self._filter.Q = Q(dt=dt, autocorrelation=Q_0)

        # Initialize the dynamics model
        self.F = G(dt)
        self.fx = lambda x: np.dot(self.F, x)
        self.FJacobian = lambda x: self.F  # noqa : ARG
        self.hx = hx
        self.HJacobian = HJacobian

        # Initialize the log innovation flag
        self.log_innovation = log_innovation
        self.code_only = code_only

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

    def _compute(self, epoch: Epoch, **kwargs) -> Series | DataFrame:
        """Computes the state vector estimate for the given epoch.

        Args:
            epoch (Epoch): Epoch object.
            **kwargs: Additional keyword arguments.

        Returns:
            Series | DataFrame: State vector estimate.
        """
        # Preprocess the data
        z, sv_pos = self._preprocess(epoch=epoch, computational_format=True, **kwargs)

        # Run the predict-update loop
        outs = self._filter.predict_update(
            x_posterior=self.x0,
            P_posterior=self.P0,
            z=z,
            fx=self.fx,
            FJacobian=self.FJacobian,
            hx=self.hx,
            HJacobian=self.HJacobian,
            fx_kwargs={},
            hx_kwargs={"sv_pos": sv_pos},
            FJ_kwargs={},
            HJ_kwargs={"sv_pos": sv_pos},
        )

        # Update the posterior estimates
        self.x0 = outs["x_posterior"].copy()
        self.P0 = outs["P_posterior"].copy()

        # Process the state
        state = self.process_state(outs["x_posterior"])

        # Log the innovation if required
        if self.log_innovation:
            state["innovation"] = outs["innovation_residual"]

        # Log the Trace of the covariance matrix
        state["P_trace"] = np.trace(self.P0)

        return state

    def fixed_interval_smoothing(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        epoches: list[Epoch],
        **kwargs,  # noqa : ARG
    ) -> tuple[DataFrame, dict[str, np.ndarray]]:
        """Computes the fixed interval smoothing estimates for the given epoches.

        Args:
            x0 (np.ndarray): Initial state vector.
            P0 (np.ndarray): Initial state covariance matrix.
            epoches (list[Epoch]): List of epoches.
            **kwargs: Additional keyword arguments.

        Returns:
           tuple[DataFrame, dict[str, np.ndarray]]: State estimates and additional outputs.

        Note:
            - The epoches must be contiguous i.e have the same satellites as the initial epoch. (See navigator.epoch.EpochCollection for more information)
        """
        # Get the initial sv_map
        sv_map = epoches[0].obs_data.index  # Contains the name of the satellites

        # Process all the epoches to get a timeseries of measurements and sv positions
        meas = [
            self._preprocess(
                epoch=epoch,
                computational_format=True,
                sv_filter=sv_map,
                code_only=self.code_only,
            )
            for epoch in epoches
        ]
        z = np.vstack([m[0] for m in meas])  # (T, 2 * num_sv)
        sv_pos = np.stack([m[1] for m in meas])  # (T, 2 * num_sv, 3)

        # Batch Process the epoches
        outs = self._filter.batch_smoothing(
            x0=x0,
            P0=P0,
            z_ts=z,
            fx=self.fx,
            FJacobian=self.FJacobian,
            hx=self.hx,
            HJacobian=self.HJacobian,
            fx_kwargs={},
            hx_kwargs={"sv_pos": sv_pos},
            FJ_kwargs={},
            HJ_kwargs={"sv_pos": sv_pos},
        )

        # Process the state estimates
        state = DataFrame(outs["x_smoothed"], columns=self.state)

        # Remove the state estimates from the outs dictionary
        outs.pop("x_smoothed")

        return state, outs

    def _dim_state(self) -> int:
        return 8
