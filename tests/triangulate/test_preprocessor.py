from navigator.core.triangulate.itriangulate.preprocessor.gps_preprocessor import (
    GPSPreprocessor,
)
import pytest
from navigator.epoch import Epoch
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

    # Do a initial profile so that no prior data is used
    epoches[0].profile = epoches[0].INITIAL
    # Calculate the pseudoranges and sat_pos
    pseudoranges, sat_pos = preprocessor(epoches[0])

    # Check that the pseudoranges and sat_pos are the correct shape
    assert isinstance(pseudoranges, pd.Series)
    assert isinstance(sat_pos, pd.DataFrame)

    # Check that the pseudoranges and sat_pos are the correct shape
    assert pseudoranges.shape[0] == sat_pos.shape[0]
