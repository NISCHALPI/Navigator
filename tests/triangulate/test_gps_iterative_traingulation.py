import pandas as pd
import pytest
from navigator.utility.epoch import Epoch
from tests.common_fixtures import nav_data, obs_data, navfilepath, obsfilepath
from navigator.satlib.triangulate import Triangulate, GPSIterativeTriangulationInterface
import numpy as np


@pytest.fixture
def epoch(
    nav_data, obs_data, navfilepath, obsfilepath
) -> tuple[list[Epoch], pd.Series, pd.Series]:
    return Epoch.epochify(obs=obs_data[1], nav=nav_data[1]), nav_data[0], obs_data[0]


def test_traingulation_gps(epoch) -> tuple[pd.DataFrame, pd.DataFrame]:
    epoches, nav, obs = epoch

    # Triangulate the epoch``
    triangulator = Triangulate(
        interface=GPSIterativeTriangulationInterface(),
    )

    # coords list
    coords = []

    # Trianguate all the epochs
    for eph in epoches:
        series = triangulator(obs=eph, obs_metadata=obs, nav_metadata=nav)
        coords.append(np.array([series["x"], series["y"], series["z"]]))

    # Take the average of the coords
    coords = np.array(coords)

    # Average the coords
    avg_coords = np.mean(coords, axis=0) / 1000

    assert avg_coords.shape == (3,)

    # Between 10 km +- of the earth radius
    assert np.linalg.norm(avg_coords) > 6371 - 10
    assert np.linalg.norm(avg_coords) < 6371 + 10

    # Real coords of YELLCAN (from NASA)
    real_coords = np.array([-1224452.4000, -2689216.0000, 5633638.2000])

    # Rescale the coords
    avg_coords = avg_coords * 1000
    # Assert the coords are close to the real coords
    assert np.linalg.norm(avg_coords - real_coords) < 100
