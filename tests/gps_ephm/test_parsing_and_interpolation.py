import pytest
from navigator.parse import Parser, IParseGPSNav
from navigator.satlib import Satellite, IGPSEphemeris
from tests.common_fixtures import nav_data, obs_data, navfilepath, obsfilepath
import numpy as np


def test_ephemeris_gps_and_sv_interpolation(navfilepath, nav_data):
    # Unpack the nav_data
    metadata, data = nav_data  # This is how the functional class is called.

    # Create an object of the functional class
    satellite = Satellite(iephemeris=IGPSEphemeris())

    # Let's take the Satellites at 2023-06-07  16:00:00 TOC (Time of Clock) to interpolate the satellite position to 19:00:00 GPS time
    final_time = "2023-07-03 18:00:00"

    # Filter the satellites that has 16:00:00 TOC.
    hr_16_data = data[data.index.get_level_values(0) == "2023-07-03 16:00:00"]

    trajectory = satellite.trajectory(
        t_sv="2023-07-03 16:00:00",  # Start time
        metadata=metadata,
        data=data,
        interval=60 * 60 * 2,  # 3 hours interval
        step=60,  # Every 60 seconds resolution
    )

    assert trajectory.shape == (23, 3, 120)

    # Get One coordinate of the satellite
    x = trajectory[0, :, 0] / 1000

    distance = np.sqrt((x**2).sum())
    assert distance == pytest.approx(26500, abs=1000), "Distance is not correct"
