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

from copy import deepcopy

import numpy as np
from filterpy.common import Q_discrete_white_noise, Saver
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from pandas.core.api import DataFrame, Series

from navigator.utility import Epoch

from ..algos.unscented_kalman_method import fx, hx
from ..itriangulate import Itriangulate

__all__ = ['UnscentedKalmanTriangulationInterface']


class UnscentedKalmanTriangulationInterface(Itriangulate):
    """Unscented Kalman Method for Triangulation."""

    def __init__(
        self,
        num_satellite: int = 5,
        dt: float = 30.0,
        simga_r: float = 6.0,
        sigma_q: float = 0.01,
        S_f: float = 36.0,
        S_g: float = 0.01,
        saver: bool = False,
    ) -> None:
        """Initializes an instance of the Unscented Kalman Method for Triangulation.

        Args:
            num_satellite (int, optional): Number of satellites to track using the UKF. Defaults to 5.
            dt (float, optional): The sampling time interval in seconds. Defaults to 30..
            simga_r (float, optional): The measurement noise for the pseudorange measurement. Defaults to 6.
            sigma_q (float, optional): The process noise for the state transition. Defaults to 0.01.
            S_f (float, optional): The white noise spectral density for the random walk clock velocity error. Defaults to 36..
            S_g (float, optional): The white noise spectral density for the random walk clock drift error. Defaults to 0.01.
            saver (bool, optional): Whether to save the intermediate results. Defaults to False.
        """
        # Check if the number of satellites to track is valid
        assert (
            num_satellite >= 4
        ), "Number of satellites to track must be greater than or equal to 4."
        self.num_satellite = num_satellite

        # Check if the sampling time interval is valid
        assert dt > 0, "Sampling time interval must be greater than 0."
        self.dt = dt

        # Check if the measurement noise is valid
        assert simga_r > 0, "Measurement noise must be greater than 0."
        self.simga_r = simga_r

        # Check if the process noise is valid
        assert sigma_q > 0, "Process noise must be greater than 0."
        self.sigma_q = sigma_q

        # Check if the white noise spectral density for the random walk clock velocity error is valid
        assert (
            S_f > 0
        ), "White noise spectral density for the random walk clock velocity error must be greater than 0."
        self.S_f = S_f

        # Check if the white noise spectral density for the random walk clock drift error is valid
        assert (
            S_g > 0
        ), "White noise spectral density for the random walk clock drift error must be greater than 0."
        self.S_g = S_g

        # Initialize the Unscented Kalman Filter
        self.ukf = UnscentedKalmanFilter(
            dim_x=8,  # Number of state variables
            dim_z=num_satellite,  # Number of measurement variables
            dt=dt,  # Sampling time interval
            fx=fx,  # State transition function
            hx=hx,  # Measurement function
            points=MerweScaledSigmaPoints(n=8, alpha=0.1, beta=2.0, kappa=-1),
        )

        # Save the intermediate results if specified
        if saver:
            self.saver = Saver(self.ukf)

        # Initialize the Unscented Kalman Filter
        self._ukf_init()  # Do not remove this line from here

        # Initialize
        super().__init__(feature="UKF Triangulation")

    def _ukf_init(self) -> None:
        """Initializes the Unscented Kalman Filter."""
        # Initialize the state vector
        self.ukf.x = np.ones(8, dtype=np.float64)

        # Initialize the state covariance matrix
        self.ukf.P = np.eye(8, dtype=np.float64) * 100

        # Initialize the process noise covariance matrix
        Q = np.kron(
            np.eye(4, dtype=np.float64),
            Q_discrete_white_noise(dim=2, dt=self.dt, var=self.sigma_q),
        )

        # Initialize the clock noise covariance matrix
        Q[-2:, -2:] = np.array(
            [[self.S_f * self.dt, 0], [0, 0]], dtype=np.float64
        ) + Q_discrete_white_noise(dim=2, dt=30, var=self.S_g)

        # Set the process noise covariance matrix
        self.ukf.Q = Q

        # Set the measurement noise covariance matrix
        self.ukf.R = np.eye(self.num_satellite, dtype=np.float64) * self.simga_r**2

    def _compute(
        self,
        obs: Epoch,
        obs_metadata: Series,
        nav_metadata: Series,
        *args,  # noqa : ARG002
        **kwargs,
    ) -> Series | DataFrame:
        """Computes the triangulated position using the Unscented Kalman Filter.

        Args:
            obs (Epoch):  The epoch to be processed.
            obs_metadata (Series): Metadata associated with the epoch.
            nav_metadata (Series): Metadata associated with the navigation data.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Series | DataFrame: A pandas series containing the triangulated position, DOPS, and clock bias and drift.
        """
        # Check if the epoch has the minimum number of satellites
        if len(obs) < self.num_satellite:
            raise ValueError(
                f"Epoch must have at least {self.num_satellite} satellites."
            )

        # Remove the extra satellites from the epoch
        copy_obs = deepcopy(obs)
        copy_obs.obs_data = copy_obs.obs_data.iloc[: self.num_satellite]
        copy_obs.trim()  # Remove the extra satellites from the navigation data

        # Get the range and sv_coordinates
        pseudorange, sv_coordinates = self._preprocess(
            epoch=copy_obs,
            obs_metadata=obs_metadata,
            nav_metadata=nav_metadata,
            **kwargs,
        )

        # Run the Unscented Kalman Predic and Update Loop
        self.ukf.predict()
        self.ukf.update(
            pseudorange.values, sv_location=sv_coordinates[["x", "y", "z"]].values
        )

        # Save the intermediate results if saver is specified
        if hasattr(self, "saver"):
            self.saver.save()

        # Return Dicts
        results = {
            "x": self.ukf.x[0],
            "y": self.ukf.x[2],
            "z": self.ukf.x[4],
            "x_dot": self.ukf.x[1],
            "y_dot": self.ukf.x[3],
            "z_dot": self.ukf.x[5],
            "cdt": self.ukf.x[6],
            "cdt_dot": self.ukf.x[7],
            "sigma_x": self.ukf.P[0, 0],
            "sigma_y": self.ukf.P[2, 2],
            "sigma_z": self.ukf.P[4, 4],
            "sigma_x_dot": self.ukf.P[1, 1],
            "sigma_y_dot": self.ukf.P[3, 3],
            "sigma_z_dot": self.ukf.P[5, 5],
            "sigma_cdt": self.ukf.P[6, 6],
            "sigma_cdt_dot": self.ukf.P[7, 7],
        }

        return Series(results)
