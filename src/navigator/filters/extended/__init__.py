"""Modules for extended filters.

This package contains numba-accelerated implementations of the extended Kalman filter (EKF). The functional 
interface contains EKF prediction and update methods, and the adaptive EKF is implemented as a subclass of the
EKF. The adaptive EKF uses the innovation covariance to adapt the process noise covariance matrix. 

Classes:
    - ExtendedKalmanFilter: Implements the standard Extended Kalman Filter (EKF) with prediction and update methods.

    - InnovationBasedAdaptiveExtendedKalmanFilter: A subclass of ExtendedKalmanFilter that adds adaptive capabilities.
      The adaptive EKF uses the innovation covariance to dynamically adjust the process noise covariance matrix during
      the filtering process.

"""
