""""Extended Kalman Filter (EKF) Implementation for Nonlinear Systems.

This module provides an implementation of the Extended Kalman Filter (EKF) designed for arbitrary nonlinear systems. 
The primary class, `ExtendedKalmanFilter`, offers a flexible interface for state estimation, leveraging Numba for enhanced computational speed.

Classes:
    ExtendedKalmanFilter: Implementation of the EKF for nonlinear systems.


Example:
    Import and Use:
    ```python
    from ekf_module import ExtendedKalmanFilter

    # Create an EKF instance for a 1D motion system
    ekf = ExtendedKalmanFilter(dim_x=2, dim_y=1)

    # Loop through time steps
    for t in range(50):
        # Use Input
        ...

        # Perform predict and update steps with EKF
        ekf.predict(F=np.eye(2), u=u)
        ekf.update(y=np.array([measurement]), HJacobian=lambda x: np.array([[1.0, 0.0]]), hx=lambda x: np.array([x[0]]))

        # Access estimated state
        estimated_state = ekf.x

        # Do something with the estimated state (e.g., store, plot, etc.)
    ```

Note:
- Detailed documentation for class methods is available within the code.
- Ensure the predict method is called before the update method for proper EKF functioning.

Author:
    Nischal Bhattarai (nischalbhattaraipi@gmail.com)
"""

import numpy as np

__all__ = [
    "ExtendedKalmanFilter",
]


class ExtendedKalmanFilter:
    """ExtendedKalmanFilter(dim_x, dim_y).

    Implementation of the Extended Kalman Filter (EKF) for nonlinear systems.

    Args:
        dim_x (int): Dimension of the state vector.
        dim_y (int): Dimension of the measurement vector.

    Attributes:
        dim_x (int): Dimension of the state vector.
        dim_y (int): Dimension of the measurement vector.

    Methods:
        predict(F: np.ndarray, u: np.ndarray = None) -> None:
            Predict step of the EKF for linear dynamic systems.

        non_linear_predict(
            fx: callable,
            FJacobian: callable,
            fx_kwargs: dict = {},
            FJacobian_kwargs: dict = {},
        ) -> None:
            Predict step of the EKF for nonlinear dynamic systems.

        update(
            y: np.ndarray,
            hx: callable,
            HJacobian: callable,
            hx_kwargs: dict = {},
            HJ_kwargs: dict = {},
        ) -> None:
            Update the state vector and the state covariance matrix.

        predict_update(
            y: np.ndarray,
            F: np.ndarray,
            HJacobian: callable,
            hx: callable,
            hx_kwargs: dict = {},
            HJ_kwargs: dict = {},
            u: np.ndarray = None,
        ) -> None:
            Perform the predict and update steps of the EKF for linear dynamic systems.

    Properties:
        x (np.ndarray): State vector getter and setter.
        P (np.ndarray): State covariance matrix getter and setter.
        F (np.ndarray): State transition matrix getter and setter.
        Q (np.ndarray): Process noise covariance matrix getter and setter.
        R (np.ndarray): Measurement noise covariance matrix getter and setter.
    """

    TERMS = {
        "PriorEstimate": "x_prior",
        "PriorCovariance": "P_prior",
        "StateJacobian": "State_Jacobian",
        "PosteriorEstimate": "x_posterior",
        "PosteriorCovariance": "P_posterior",
        "InnovationResidual": "innovation_residual",
        "InnovationCovariance": "innovation_covariance",
        "KalmanGain": "Kalman_gain",
        "MeasurementJacobian": "Measurement_Jacobian",
        "SmoothedEstimate": "x_smoothed",
        "SmoothedCovariance": "P_smoothed",
    }

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
    ) -> None:
        """Initialize the EKF.

        Args:
            dim_x (int): Dimension of the state vector
            dim_z (int): Dimension of the measurement vector


        Raises:
            ValueError: If the state vector dimension or the measurement vector dimension is less than 1.

        Note:
            - The state vector is initialized to a zero vector.
            - The state covariance matrix is initialized to an identity matrix.
            - The process noise covariance matrix is initialized to a zero matrix.
            - The measurement noise covariance matrix is initialized to an identity matrix.


        Returns:
            None
        """
        # Store the dimensions of the state vector and the measurement vector
        if dim_x < 1:
            raise ValueError("State vector dimension must be greater than 0")
        if dim_z < 1:
            raise ValueError("Measurement vector dimension must be greater than 0")
        self.dim_x = dim_x
        self.dim_y = dim_z

        # Initialize the process noise covariance matrix
        self._Q = np.zeros((dim_x, dim_x), dtype=np.float64)
        # Initialize the measurement noise covariance matrix
        self._R = np.eye(dim_z, dtype=np.float64)

    def predict(
        self,
        x_posterior: np.ndarray,
        P_posterior: np.ndarray,
        fx: callable,
        FJacobian: callable,
        fx_kwargs: dict = {},
        FJacobian_kwargs: dict = {},
    ) -> dict[str, np.ndarray]:
        """Predict step of the Extended Kalman Filter (EKF) for nonlinear dynamic systems.

        Args:
            x_posterior (np.ndarray): Posterior state vector. (dim_x,)
            P_posterior (np.ndarray): Posterior state covariance matrix. (dim_x, dim_x)
            fx (callable): Nonlinear state transition function. (dim_x,) -> (dim_x,)
            FJacobian (callable): Jacobian matrix calculation function for the nonlinear state transition function. (dim_x,) -> (dim_x, dim_x)
            fx_kwargs (dict): Keyword arguments for the nonlinear state transition function. Default is an empty dictionary.
            FJacobian_kwargs (dict): Keyword arguments for the Jacobian matrix of the nonlinear state transition function. Default is an empty dictionary.

        Required Function Signature:
            fx : def fx(x: np.ndarray, **kwargs) -> np.ndarray
            FJacobian : def FJacobian(x: np.ndarray, **kwargs) -> np.ndarray

            Warning : All additional keyword arguments must be named arguments and must be passed as a dictionary.

            Note : The control input vector is passed as a keyword argument to the state transition function as
            a named argument which is controlled by the user.

        Returns:
            dict[str, np.ndarray]: The predicted state vector, the predicted state covariance matrix, and the Jacobian matrix of the nonlinear state transition function
        """
        # Compute the predicted state vector
        x_prior = fx(x_posterior, **fx_kwargs)

        # Compute the Jacobian matrix of the nonlinear state transition function
        F = FJacobian(x_posterior, **FJacobian_kwargs)

        # Compute the predicted state covariance matrix
        P_prior = F @ P_posterior @ F.T + self._Q

        return {
            self.TERMS["PriorEstimate"]: x_prior,
            self.TERMS["PriorCovariance"]: P_prior,
            self.TERMS["StateJacobian"]: F,
        }

    def update(
        self,
        z: np.ndarray,
        x_prior: np.ndarray,
        P_prior: np.ndarray,
        hx: callable,
        HJacobian: callable,
        hx_kwargs: dict = {},
        HJ_kwargs: dict = {},
    ) -> dict[str, np.ndarray]:
        """Update the state vector and the state covariance matrix.

        Args:
            z (np.ndarray): Measurement vector. (dim_z,)
            x_prior (np.ndarray): Prior state vector. (dim_x,)
            P_prior (np.ndarray): Prior state covariance matrix. (dim_x, dim_x)
            hx (callable): Nonlinear measurement function. (dim_x,) -> (dim_z,)
            HJacobian (callable): Jacobian matrix calculation function for the nonlinear measurement function. (dim_x,) -> (dim_z, dim_x)
            hx_kwargs (dict): Keyword arguments for the nonlinear measurement function. Default is an empty dictionary.
            HJ_kwargs (dict): Keyword arguments for the Jacobian matrix of the nonlinear measurement function. Default is an empty dictionary.

        Required Function Signature:
            hx : def hx(x: np.ndarray, **kwargs) -> np.ndarray
            HJacobian : def HJacobian(x: np.ndarray, **kwargs) -> np.ndarray

        Warnings:
            - All additional keyword arguments must be named arguments and must be passed as a dictionary.
            - This method assumes that the predict method has been called before this method. EKF wiil not function properly if the predict method is not called before this method.

        Returns:
            float : The innovation residual
        """
        # Check if the measurement vector is of the correct dimension
        if z.shape != (self.dim_y,):
            raise ValueError(
                f"Measurement vector must be of shape {self.dim_y} but is {z.shape}"
            )

        # Compute the innovation residual
        y = z - hx(x_prior, **hx_kwargs)

        # Compute the Jacobian matrix of the nonlinear measurement function
        H = HJacobian(x_prior, **HJ_kwargs)

        # Compute the innovation covariance matrix
        S = H @ P_prior @ H.T + self._R
        # Compute the Kalman gain
        K = P_prior @ H.T @ np.linalg.inv(S)

        # Update the state vector
        x_post = x_prior + K @ y
        # Update the state covariance matrix using joseph form
        factor = np.eye(self.dim_x, dtype=np.float64) - K @ H
        P_post = factor @ P_prior @ factor.T + K @ self._R @ K.T

        return {
            self.TERMS["PosteriorEstimate"]: x_post,
            self.TERMS["PosteriorCovariance"]: P_post,
            self.TERMS["InnovationResidual"]: y,
            self.TERMS["InnovationCovariance"]: S,
            self.TERMS["KalmanGain"]: K,
            self.TERMS["MeasurementJacobian"]: H,
        }

    def predict_update(
        self,
        x_posterior: np.ndarray,
        P_posterior: np.ndarray,
        z: np.ndarray,
        fx: callable,
        FJacobian: callable,
        hx: callable,
        HJacobian: callable,
        fx_kwargs: dict = {},
        hx_kwargs: dict = {},
        FJ_kwargs: dict = {},
        HJ_kwargs: dict = {},
    ) -> dict[str, np.ndarray]:
        """Perform the predict and update steps of the EKF for nonlinear dynamic systems.

        Args:
            x_posterior (np.ndarray): Posterior state vector. (dim_x,)
            P_posterior (np.ndarray): Posterior state covariance matrix. (dim_x, dim_x)
            z (np.ndarray): Measurement vector. (dim_z,)
            fx (callable): Nonlinear state transition function. (dim_x,) -> (dim_x,)
            FJacobian (callable): Jacobian matrix calculation function for the nonlinear state transition function. (dim_x,) -> (dim_x, dim_x)
            hx (callable): Nonlinear measurement function. (dim_x,) -> (dim_z,)
            HJacobian (callable): Jacobian matrix calculation function for the nonlinear measurement function. (dim_x,) -> (dim_z, dim_x)
            fx_kwargs (dict): Keyword arguments for the nonlinear state transition function. Default is an empty dictionary.
            FJ_kwargs (dict): Keyword arguments for the Jacobian matrix of the nonlinear state transition function. Default is an empty dictionary.
            hx_kwargs (dict): Keyword arguments for the nonlinear measurement function. Default is an empty dictionary.
            HJ_kwargs (dict): Keyword arguments for the Jacobian matrix of the nonlinear measurement function. Default is an empty dictionary.

        Required Function Signature:
            fx : def fx(x: np.ndarray, **kwargs) -> np.ndarray
            FJacobian : def FJacobian(x: np.ndarray, **kwargs) -> np.ndarray
            hx : def hx(x: np.ndarray, **kwargs) -> np.ndarray
            HJacobian : def HJacobian(x: np.ndarray, **kwargs) -> np.ndarray

        Warnings:
            - All additional keyword arguments must be named arguments and must be passed as a dictionary.
            - This method assumes that the predict method has been called before this method. EKF wiil not function properly if the predict method is not called before this method.

        Returns:
            float : The innovation residual
        """
        # Perform the predict step
        prediction = self.predict(
            x_posterior=x_posterior,
            P_posterior=P_posterior,
            fx=fx,
            fx_kwargs=fx_kwargs,
            FJacobian=FJacobian,
            FJacobian_kwargs=FJ_kwargs,
        )

        # Perform the update step
        update = self.update(
            z,
            prediction[self.TERMS["PriorEstimate"]],
            prediction[self.TERMS["PriorCovariance"]],
            hx,
            HJacobian,
            hx_kwargs,
            HJ_kwargs,
        )

        return {**prediction, **update}

    def fixed_interval_smoothing(
        self,
        x_posterior: np.ndarray,
        P_posterior: np.ndarray,
        FJacobian: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Fixed Interval Smoothing for Nonlinear Systems using the Extended Kalman Filter (EKF).

        Args:
            x_posterior (np.ndarray): Posterior state vector timeseries. (T, dim_x)
            P_posterior (np.ndarray): Posterior state covariance matrix timeseries. (T, dim_x, dim_x)
            FJacobian (np.ndarray): Jacobian matrix of the nonlinear state transition function timeseries. (T, dim_x, dim_x)

        Returns:
            dict[str, np.ndarray]: Smoothed state vector timeseries and smoothed state covariance matrix timeseries
        """
        # Initialize the smoothed state vector and the smoothed state covariance matrix
        x_smoothed = np.zeros_like(x_posterior)
        P_smoothed = np.zeros_like(P_posterior)

        # Since last state is already the smoothed state
        # Add the last state to the smoothed state vector
        x_smoothed[-1] = x_posterior[-1]
        P_smoothed[-1] = P_posterior[-1]

        # Start from the second last state and go back in time
        T = x_posterior.shape[0]

        for t in range(T - 2, -1, -1):
            # Caclulate the prior covariance matrix i.e P_{t+1|t}
            P_prior = FJacobian[t + 1] @ P_posterior[t] @ FJacobian[t + 1].T + self._Q
            # Compute the Smoothing Gain
            L = P_posterior[t] @ FJacobian[t + 1].T @ np.linalg.inv(P_prior)

            # Compute the smoothed state vector
            x_smoothed[t] = x_posterior[t] + L @ (
                x_smoothed[t + 1] - FJacobian[t + 1] @ x_posterior[t]
            )
            # Compute the smoothed state covariance matrix
            P_smoothed[t] = P_posterior[t] + L @ (P_smoothed[t + 1] - P_prior) @ L.T

        return {
            self.TERMS["SmoothedEstimate"]: x_smoothed,
            self.TERMS["SmoothedCovariance"]: P_smoothed,
        }

    def batch_filtering(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        z_ts: np.ndarray,
        fx: callable,
        FJacobian: callable,
        hx: callable,
        HJacobian: callable,
        fx_kwargs: dict = {},
        FJ_kwargs: dict = {},
        hx_kwargs: dict = {},
        HJ_kwargs: dict = {},
    ) -> dict[str, np.ndarray]:
        """Batch Filtering for Nonlinear Systems using the Extended Kalman Filter (EKF).

        Args:
            x0 (np.ndarray): Initial state vector. (dim_x,)
            P0 (np.ndarray): Initial state covariance matrix. (dim_x, dim_x)
            z_ts (np.ndarray): Measurement vector timeseries. (T, dim_z)
            fx (callable): Nonlinear state transition function. (dim_x,) -> (dim_x,)
            FJacobian (callable): Jacobian matrix calculation function for the nonlinear state transition function. (dim_x,) -> (dim_x, dim_x)
            hx (callable): Nonlinear measurement function. (dim_x,) -> (dim_z,)
            HJacobian (callable): Jacobian matrix calculation function for the nonlinear measurement function. (dim_x,) -> (dim_z, dim_x)
            fx_kwargs (dict): Keyword arguments for the nonlinear state transition function. Default is an empty dictionary.
            FJ_kwargs (dict): Keyword arguments for the Jacobian matrix of the nonlinear state transition function. Default is an empty dictionary.
            hx_kwargs (dict): Keyword arguments for the nonlinear measurement function. Default is an empty dictionary.
            HJ_kwargs (dict): Keyword arguments for the Jacobian matrix of the nonlinear measurement function. Default is an empty dictionary.

        Returns:
            dict[str, np.ndarray]: Filtered state vector timeseries, filtered state covariance matrix timeseries, and innovation residuals timeseries

        Note:
            - The keywords arguments must have a time dimension of T so that the arguments can be passed to the functions as a timeseries.
        """
        # Initialize the state vector and the state covariance matrix timeseries
        outs = {
            term: []
            for term in self.TERMS.values()
            if term
            not in [self.TERMS["SmoothedEstimate"], self.TERMS["SmoothedCovariance"]]
        }

        # Run the EKF for each time step
        for t in range(z_ts.shape[0]):
            # Perform the predictupdate step of the EKF
            results = self.predict_update(
                x_posterior=(
                    x0 if t == 0 else outs[self.TERMS["PosteriorEstimate"]][t - 1]
                ),
                P_posterior=(
                    P0 if t == 0 else outs[self.TERMS["PosteriorCovariance"]][t - 1]
                ),
                z=z_ts[t],
                fx=fx,
                FJacobian=FJacobian,
                hx=hx,
                HJacobian=HJacobian,
                fx_kwargs={k: v[t] for k, v in fx_kwargs.items()},
                FJ_kwargs={k: v[t] for k, v in FJ_kwargs.items()},
                hx_kwargs={k: v[t] for k, v in hx_kwargs.items()},
                HJ_kwargs={k: v[t] for k, v in HJ_kwargs.items()},
            )

            # Append the results to the timeseries
            for key, value in results.items():
                outs[key].append(value)

        # Stack the lists to form the timeseries
        for key, value in outs.items():
            outs[key] = np.stack(value)

        return outs

    def batch_smoothing(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        z_ts: np.ndarray,
        fx: callable,
        FJacobian: callable,
        hx: callable,
        HJacobian: callable,
        fx_kwargs: dict = {},
        FJ_kwargs: dict = {},
        hx_kwargs: dict = {},
        HJ_kwargs: dict = {},
    ) -> dict[str, np.ndarray]:
        """Batch Smoothing for Nonlinear Systems using the Extended Kalman Filter (EKF).

        Args:
            x0 (np.ndarray): Initial state vector. (dim_x,)
            P0 (np.ndarray): Initial state covariance matrix. (dim_x, dim_x)
            z_ts (np.ndarray): Measurement vector timeseries. (T, dim_z)
            fx (callable): Nonlinear state transition function. (dim_x,) -> (dim_x,)
            FJacobian (callable): Jacobian matrix calculation function for the nonlinear state transition function. (dim_x,) -> (dim_x, dim_x)
            hx (callable): Nonlinear measurement function. (dim_x,) -> (dim_z,)
            HJacobian (callable): Jacobian matrix calculation function for the nonlinear measurement function. (dim_x,) -> (dim_z, dim_x)
            fx_kwargs (dict): Keyword arguments for the nonlinear state transition function. Default is an empty dictionary.
            FJ_kwargs (dict): Keyword arguments for the Jacobian matrix of the nonlinear state transition function. Default is an empty dictionary.
            hx_kwargs (dict): Keyword arguments for the nonlinear measurement function. Default is an empty dictionary.
            HJ_kwargs (dict): Keyword arguments for the Jacobian matrix of the nonlinear measurement function. Default is an empty dictionary.

        Returns:
            dict[str, np.ndarray]: Smoothed state vector timeseries, smoothed state covariance matrix timeseries, and innovation residuals timeseries

        Note:
            - The keywords arguments must have a time dimension of T so that the arguments can be passed to the functions as a timeseries.
        """
        # Perform the batch filtering
        outs = self.batch_filtering(
            x0=x0,
            P0=P0,
            z_ts=z_ts,
            fx=fx,
            FJacobian=FJacobian,
            hx=hx,
            HJacobian=HJacobian,
            fx_kwargs=fx_kwargs,
            FJ_kwargs=FJ_kwargs,
            hx_kwargs=hx_kwargs,
            HJ_kwargs=HJ_kwargs,
        )

        # Update the outputs with the smoothed estimates
        outs.update(
            self.fixed_interval_smoothing(
                x_posterior=outs[self.TERMS["PosteriorEstimate"]],
                P_posterior=outs[self.TERMS["PosteriorCovariance"]],
                FJacobian=outs[self.TERMS["StateJacobian"]],
            )
        )
        return outs

    @property
    def Q(self) -> np.ndarray:
        """Return the process noise covariance matrix.

        Returns:
            np.ndarray: Process noise covariance matrix
        """
        return self._Q

    @property
    def R(self) -> np.ndarray:
        """Return the measurement noise covariance matrix.

        Returns:
            np.ndarray: Measurement noise covariance matrix
        """
        return self._R

    @Q.setter
    def Q(self, Q: np.ndarray) -> None:
        """Set the process noise covariance matrix.

        Args:
            Q (np.ndarray): Process noise covariance matrix
        """
        if Q.shape != (self.dim_x, self.dim_x):
            raise ValueError(
                f"Process noise covariance matrix must be of shape {self._Q.shape} but is {Q.shape}"
            )

        # Set the process noise covariance matrix
        self._Q = Q

    @R.setter
    def R(self, R: np.ndarray) -> None:
        """Set the measurement noise covariance matrix.

        Args:
            R (np.ndarray): Measurement noise covariance matrix
        """
        if R.shape != (self.dim_y, self.dim_y):
            raise ValueError(
                f"Measurement noise covariance matrix must be of shape {self._R.shape} but is {R.shape}"
            )

        # Set the measurement noise covariance matrix
        self._R = R
