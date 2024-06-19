from navigator.core.triangulate.itriangulate.preprocessor.gps_preprocessor import (
    GPSPreprocessor,
)
import pytest
from navigator.epoch import Epoch
from navigator.epoch.loaders import from_rinex_files
from pathlib import Path
import pandas as pd
from tests.common_fixtures import navfilepath, obsfilepath


@pytest.fixture
def epoch(navfilepath, obsfilepath) -> list[Epoch]:
    return list(
        from_rinex_files(
            Path(obsfilepath),
            Path(navfilepath),
            station_name="AMC400USA",
            column_mapper={k: k for k in Epoch.OBSERVABLES},
        )
    )


def test_gps_preprocessor(epoch):
    epoches = list(epoch)

    # Preprocess the epoch
    preprocessor = GPSPreprocessor()

    # Do a initial profile so that no prior data is used
    epoches[0].profile = Epoch.INITIAL
    # Calculate the pseudoranges and sat_pos
    pseudoranges, sat_pos = preprocessor(epoches[0])

    # Check that the pseudoranges and sat_pos are the correct shape
    assert isinstance(pseudoranges, pd.DataFrame)
    assert isinstance(sat_pos, pd.DataFrame)
    print(sat_pos.columns)
    # Check that sat_pos has the correct columns
    assert all(
        col in sat_pos.columns
        for col in ["x", "y", "z", "SVclockBias", "elevation", "azimuth"]
    )

    # Check that the pseudoranges and sat_pos are the correct shape
    assert pseudoranges.shape[0] == sat_pos.shape[0]
