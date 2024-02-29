"""Phase Based Measurement Model for Precise Point Positioning (PPP) Modes.

This module defines a measurement function for Precise Point Positioning (PPP) modes
using a Kalman Filter. The function computes [code, phase] measurements for GPS PPP modes.

The measurement model is specifically tailored for the Global Positioning System (GPS)
and does not support multi-GNSS models.

State Definition:
    The state vector (curr_state) represents the receiver and is defined as follows:
    
    x = [x, x_dot, y, y_dot, z, z_dot, clock_drift, clock_drift_rate, wet_tropospheric_delay, B1, ..., Bn]

    Where:
    - x : x-coordinate of the receiver.
    - x_dot : Velocity of the x-coordinate.
    - y : y-coordinate of the receiver.
    - y_dot : Velocity of the y-coordinate.
    - z : z-coordinate of the receiver.
    - z_dot : Velocity of the z-coordinate.
    - clock_drift: Clock drift.
    - clock_drift_rate: Clock drift rate.
    - wet_tropospheric_delay: Wet tropospheric delay.
    - B: Bias of the phase measurements, including integer ambiguity and hardware delay.

Usage:
    This module is part of a larger system and provides the measurement function
    for PPP modes. It utilizes coordinate transformations, tropospheric corrections,
    and Kalman filtering to calculate accurate [code, phase] measurements.

See Also:
    - Linear observation model for PPP:
      https://gssc.esa.int/navipedia/index.php?title=Linear_observation_model_for_PPP

"""

import numpy as np
import pandas as pd

from ........utility.transforms.coordinate_transforms import geocentric_to_ellipsoidal
from .....algos.troposphere.tropospheric_delay import filtered_troposphere_correction

__all__ = ["phase_measurement_model"]


def phase_measurement_model(
    state: np.ndarray, sv_frame: pd.DataFrame, num_sv: int, day_of_year: int
) -> np.ndarray:
    """Measurement Function for the phase based GPS Kalman Filter.

    Args:
        state: The current state of the system.
        sv_frame: The satellite frame containing the satellite positions, elevation, and azimuth.
        num_sv: The number of satellites to continuously track.
        day_of_year: The day of the year.

    Returns:
        Returns: The stacked [code, phase] measurements. The code is first, and the phase is second.
    """
    # Check of the state vector is valid
    if state.size != 9 + num_sv:
        raise ValueError(
            "The state vector is invalid. It should be of size 9 + num_sv."
        )

    position = state[0:6:2]
    lat, _, height = geocentric_to_ellipsoidal(*position)

    # Get the sv positions
    sv_positions = sv_frame[["x", "y", "z"]].values

    # Check if elevation is available
    if "elevation" not in sv_frame.columns:
        raise ValueError("Elevation is required for the measurement model.")
    sv_elevation = sv_frame["elevation"].values

    # Get the current state variables
    cdt = state[6]  # cdt
    t_wet = state[8]  # t_wet
    integer_bais_vector = state[9 : 9 + num_sv]  # lambda_N

    # Compute the range from the user to the satellites
    range_from_user = np.power((sv_positions - position), 2).sum(axis=1) ** 0.5
    # Compute the tropospheric wet delay
    tropo = np.array(
        [
            filtered_troposphere_correction(
                latitude=lat,
                elevation=elev,
                height=height,
                estimated_wet_delay=t_wet,
                day_of_year=day_of_year,
            )
            for elev in sv_elevation
        ],
        dtype=np.float64,
    )

    # Get the current range from the code measurements
    phase_measurements = (
        range_from_user + cdt + tropo + integer_bais_vector
    )  # phase_measurements
    code_measurements = range_from_user + cdt + tropo

    # Return the measurements
    return np.concatenate((code_measurements, phase_measurements))


# name: src/navigator/core/triangulate/itriangulate/kalman/tools/phase_based/measurement.py
