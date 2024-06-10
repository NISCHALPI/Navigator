import pandas as pd
import pytest
from navigator.epoch import Epoch
from tests.common_fixtures import nav_data, obs_data, navfilepath, obsfilepath
from navigator.core.triangulate import Triangulate, IterativeTriangulationInterface
import numpy as np
from pathlib import Path


@pytest.fixture
def epoch(navfilepath, obsfilepath) -> list[Epoch]:
    return Epoch.epochify(Path(obsfilepath), Path(navfilepath), column_map={k :k for k in Epoch.OBSERVABLES})


def test_traingulation_gps(epoch):
    epoches = list(epoch)

    # Triangulate the epoch``
    triangulator = Triangulate(
        interface=IterativeTriangulationInterface(),
    )

    # coords list
    coords = []
    print(epoches[0].obs_data)
    # Trianguate all the epochs
    df = triangulator.triangulate_time_series(epoches)

    coords = df[["x", "y", "z"]].values

    # Take the average of the coords
    coords = np.array(coords)

    # Average the coords
    avg_coords = np.mean(coords, axis=0) / 1000

    assert avg_coords.shape == (3,)

    # Between 10 km +- of the earth radius
    assert np.linalg.norm(avg_coords) > 6371 - 10
    assert np.linalg.norm(avg_coords) < 6371 + 10

    # Real coords of AMC400USA_R_20230391700_01H_30S_MO.crx
    real_coords = np.array([-1248596.405, -4819428.21, 3976505.93])

    # Rescale the coords
    avg_coords = avg_coords * 1000
    # Assert the coords are close to the real coords
    assert np.linalg.norm(avg_coords - real_coords) < 50
