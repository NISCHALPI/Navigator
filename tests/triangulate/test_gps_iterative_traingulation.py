import pandas as pd
import pytest
from navigator.epoch import Epoch
from tests.common_fixtures import nav_data, obs_data, navfilepath, obsfilepath
from navigator.core.triangulate import Triangulate, IterativeTriangulationInterface
import numpy as np
from pathlib import Path


@pytest.fixture
def epoch(navfilepath, obsfilepath) -> list[Epoch]:
    return Epoch.epochify(Path(obsfilepath), Path(navfilepath))


def test_traingulation_gps(epoch):
    epoches = list(epoch)

    # Triangulate the epoch``
    triangulator = Triangulate(
        interface=IterativeTriangulationInterface(),
    )

    # coords list
    coords = []

    # Trianguate all the epochs
    prior = triangulator(epoch=epoches[0], apply_tropo=False, apply_iono=False)
    for eph in epoches:
        series = triangulator(
            epoch=eph,
            apply_tropo=True,
            apply_iono=True,
            prior=prior,
        )
        prior = series
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
