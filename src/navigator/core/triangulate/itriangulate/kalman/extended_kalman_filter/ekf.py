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

from .ekf_functional_interface import ekf_predict_covariance_update, ekf_update

__all__ = ["ExtendedKalmanFilter"]


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

    def __init__(self, dim_x: int, dim_y: int) -> None:
        """Initialize the EKF.

        Args:
            dim_x (int): Dimension of the state vector
            dim_y (int): Dimension of the measurement vector
        """
        # Store the dimensions of the state vector and the measurement vector
        if dim_x < 1:
            raise ValueError("State vector dimension must be greater than 0")
        if dim_y < 1:
            raise ValueError("Measurement vector dimension must be greater than 0")
        self.dim_x = dim_x
        self.dim_y = dim_y

        # Initialize the state vector
        self._x = np.zeros(dim_x)
        # Initialize the state covariance matrix
        self._P = np.eye(dim_x)
        # Initialize the process noise covariance matrix
        self._Q = np.zeros((dim_x, dim_x))
        # Initialize the measurement noise covariance matrix
        self._R = np.eye(dim_y)

        # Set a posteriori state estimate
        self._x_post = np.zeros(dim_x)
        # Set a priori state estimate
        self._x_prior = np.zeros(dim_x)

    def predict(self, F: np.ndarray, u: np.ndarray = None) -> None:
        """Predict step of the Extended Kalman Filter (EKF) for linear dynamic systems.

        Args:
            F (np.ndarray): State transition matrix.
            u (np.ndarray): Control input vector. Default is None.

        Returns:
            None
        """
        # Predict the state vector
        self._x_prior = F @ self._x_post

        if u is not None:
            self._x_prior += u

        # Predict the state covariance matrix
        self._P = ekf_predict_covariance_update(
            F.astype(np.float64), self._P.astype(np.float64), self._Q.astype(np.float64)
        )

    def non_linear_predict(
        self,
        fx: callable,
        FJacobian: callable,
        fx_kwargs: dict = {},
        FJacobian_kwargs: dict = {},
    ) -> None:
        """Predict step of the Extended Kalman Filter (EKF) for nonlinear dynamic systems.

        Args:
            fx (callable): Nonlinear state transition function.
            FJacobian (callable): Jacobian matrix calculation function for the nonlinear state transition function.
            fx_kwargs (dict): Keyword arguments for the nonlinear state transition function. Default is an empty dictionary.
            FJacobian_kwargs (dict): Keyword arguments for the Jacobian matrix of the nonlinear state transition function. Default is an empty dictionary.

        Required Function Signature:
            fx : def fx(x: np.ndarray, **kwargs) -> np.ndarray
            FJacobian : def FJacobian(x: np.ndarray, **kwargs) -> np.ndarray

            Warning : All additional keyword arguments must be named arguments and must be passed as a dictionary.

            Note : The control input vector is passed as a keyword argument to the state transition function as
            a named argument which is controlled by the user.

        Returns:
            None
        """
        # Compute the predicted state vector
        self._x_prior = fx(self._x_post, **fx_kwargs)

        # Compute the state transition matrix
        F = FJacobian(self._x_post, **FJacobian_kwargs)

        # Update the state covariance matrix
        self._P = ekf_predict_covariance_update(
            F.astype(np.float64), self._P.astype(np.float64), self._Q.astype(np.float64)
        )

    def update(
        self,
        y: np.array,
        hx: callable,
        HJacobian: callable,
        hx_kwargs: dict,
        HJ_kwargs: dict,
    ) -> None:
        """Update the state vector and the state covariance matrix.

        Args:
            y (np.ndarray): Measurement vector.
            hx (callable): Nonlinear measurement function.
            HJacobian (callable): Jacobian matrix calculation function for the nonlinear measurement function.
            hx_kwargs (dict): Keyword arguments for the nonlinear measurement function. Default is an empty dictionary.
            HJ_kwargs (dict): Keyword arguments for the Jacobian matrix of the nonlinear measurement function. Default is an empty dictionary.

        Required Function Signature:
            hx : def hx(x: np.ndarray, **kwargs) -> np.ndarray
            HJacobian : def HJacobian(x: np.ndarray, **kwargs) -> np.ndarray

        Warnings:
            - All additional keyword arguments must be named arguments and must be passed as a dictionary.
            - This method assumes that the predict method has been called before this method. EKF wiil not function properly if the predict method is not called before this method.

        Returns:
            None
        """
        # Compute the measurement residual
        y = y.flatten()
        y = y - hx(self._x_prior, **hx_kwargs)

        # Compute the measurement residual covariance
        H = HJacobian(self._x_prior, **HJ_kwargs)

        self._x_post, self._P = ekf_update(
            y_hat=y.astype(np.float64),
            x_prior=self._x_prior.astype(np.float64),
            P_prior=self._P.astype(np.float64),
            H=H.astype(np.float64),
            R=self._R.astype(np.float64),
        )

        # Set the state vector
        self._x = np.copy(self._x_post)  # Set a state

    def predict_update(
        self,
        y: np.array,
        F: np.ndarray,
        HJacobian: callable,
        hx: callable,
        hx_kwargs: dict = {},
        HJ_kwargs: dict = {},
        u: np.ndarray = None,
    ) -> None:
        """Perform the predict and update steps of the Extended Kalman Filter (EKF) for linear dynamic systems.

        Args:
            y (np.ndarray): Measurement vector.
            F (np.ndarray): State transition matrix.
            HJacobian (callable): Jacobian matrix calculation function for the nonlinear measurement function.
            hx (callable): Nonlinear measurement function.
            hx_kwargs (dict): Keyword arguments for the nonlinear measurement function. Default is an empty dictionary.
            HJ_kwargs (dict): Keyword arguments for the Jacobian matrix of the nonlinear measurement function. Default is an empty dictionary.
            u (np.ndarray): Control input vector. Default is None.

        Required Function Signature:
            hx : def hx(x: np.ndarray, **kwargs) -> np.ndarray
            HJacobian : def HJacobian(x: np.ndarray, **kwargs) -> np.ndarray

        Warnings:
            - All additional keyword arguments must be named arguments and must be passed as a dictionary.
            - This method assumes that the predict method has been called before this method. EKF wiil not function properly if the predict method is not called before this method.

        Returns:
            None
        """
        # Perform the predict step
        self.predict(F=F, u=u)

        # Perform the update step
        self.update(
            y=y, hx=hx, HJacobian=HJacobian, hx_kwargs=hx_kwargs, HJ_kwargs=HJ_kwargs
        )

    @property
    def x(self) -> np.ndarray:
        """Return the state vector.

        Returns:
            np.ndarray: State vector
        """
        return self._x

    @property
    def P(self) -> np.ndarray:
        """Return the state covariance matrix.

        Returns:
            np.ndarray: State covariance matrix
        """
        return self._P

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

    @x.setter
    def x(self, x: np.ndarray) -> None:
        """Set the state vector.

        Args:
            x (np.ndarray): State vector
        """
        if x.shape != (self.dim_x,):
            raise ValueError(
                f"State vector must be of shape {self._x.shape} but is {x.shape}"
            )

        # Set the state vector
        self._x = x
        # Set a posteriori state estimate since this is triggered while
        # initializing the state vector
        self._x_post = np.copy(self._x)

    @P.setter
    def P(self, P: np.ndarray) -> None:
        """Set the state covariance matrix.

        Args:
            P (np.ndarray): State covariance matrix
        """
        if P.shape != (self.dim_x, self.dim_x):
            raise ValueError(
                f"State covariance matrix must be of shape {self._P.shape} but is {P.shape}"
            )

        # Set the state covariance matrix
        self._P = P

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
