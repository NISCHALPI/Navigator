"""This module contains the functions for Phased Based GPS Positioning Kalman Filter.

This model is specifically tailored for the Global Positioning System (GPS) phase based model and
does not support multi-GNSS scenarios.

This model parameterizes coordinates in the ellipsoidal state space.

State Definition:
    The state vector (curr_state) represents the receiver and is defined as follows:
    
    x = [latitude, latitude_velocity, longitude, longitude_velocity, height, height_velocity, clock_drift, clock_drift_rate, wet_tropospheric_delay, B1, ..., Bn]

    Where:
    - latitude: Latitude of the receiver.
    - latitude_velocity: Velocity of the latitude.
    - longitude: Longitude of the receiver.
    - longitude_velocity: Velocity of the longitude.
    - height: Height of the receiver.
    - height_velocity: Velocity of the height.
    - clock_drift: Clock drift.
    - clock_drift_rate: Clock drift rate.
    - wet_tropospheric_delay: Wet tropospheric delay.
    - B: Bias of the phase measurements, including integer ambiguity and hardware delay.

Usage:
There are function provided for each of the following:
    - Measurement Model
        The `phase_measurement_model` function provides the measurement model for the phase based GPS Kalman Filter.
    - Process Noise Profile for Phase Based GPS Kalman Filter. 
        The `phase_process_noise_profile` function provides the process noise profile for the phase based GPS Kalman Filter.
    - State Transition Matrix for Phase Based GPS Kalman Filter. 
        The `phase_state_transistion_matrix` function provides the state transition matrix for the phase based GPS Kalman Filter.


See Also:
For more information on PPP, refer to: <https://gssc.esa.int/navipedia/index.php?title=Precise_Point_Positioning_(PPP)>
"""

from .measurement import phase_measurement_model
from .noise import phase_process_noise_profile
from .state_transistion import phase_state_transistion_matrix
