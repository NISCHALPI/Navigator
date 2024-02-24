"""Import the Kalman filters implemented in this package.

This package contains numba-accelerated implementations of the extended Kalman filter (EKF) and its adaptive variant.
TODO: Add more filters to this package.

Example:
    ```python
    from navigator.filters import ExtendedKalmanFilter, InnovationBasedAdaptiveExtendedKalmanFilter

    # Now you can use the imported classes in your code
    ekf_instance = ExtendedKalmanFilter()
    adaptive_ekf_instance = InnovationBasedAdaptiveExtendedKalmanFilter()
    ```


"""

from .extended.ekf import (
    ExtendedKalmanFilter,
    InnovationBasedAdaptiveExtendedKalmanFilter,
)
