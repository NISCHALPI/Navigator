"""This module contains the Extended Kalman Filter (EKF) implementation for the GPS Kalman filter problem.

The EKF is a nonlinear version of the Kalman filter that linearizes the measurement and state transition functions using the Jacobian matrix.
"""
from .ekf import ExtendedKalmanFilter
