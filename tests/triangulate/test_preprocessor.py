from navigator.satlib.triangulate.itriangulate.preprocess import GPSPreprocessor
import pytest
from navigator.utility import Epoch
from pathlib import Path
import pandas as pd
from tests.common_fixtures import navfilepath, obsfilepath


@pytest.fixture
def epoch(navfilepath, obsfilepath) -> list[Epoch]:
    return Epoch.epochify(Path(obsfilepath), Path(navfilepath))


def test_gps_preprocessor(epoch):
    epoches = list(epoch)

    # Preprocess the epoch
    preprocessor = GPSPreprocessor()

    # Calculate the pseudoranges and sat_pos
    pseudoranges, sat_pos = preprocessor(
        epoches[0], apply_tropo=False, apply_iono=False
    )

    # Check that the pseudoranges and sat_pos are the correct shape
    assert isinstance(pseudoranges, pd.Series)
    assert isinstance(sat_pos, pd.DataFrame)

    # Check that the pseudoranges and sat_pos are the correct shape
    assert pseudoranges.shape[0] == sat_pos.shape[0]
