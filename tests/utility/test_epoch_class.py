import pandas as pd
import pytest
from navigator.epoch import Epoch
from navigator.epoch.loaders import from_rinex_files
from tests.common_fixtures import nav_data, obs_data, navfilepath, obsfilepath
from pathlib import Path


def test_epochify(obsfilepath, navfilepath):
    """Test the epochify function"""
    obs_path = Path(obsfilepath)
    nav_path = Path(navfilepath)

    # Epochify the data
    epochified = list(
        from_rinex_files(
            observation_file=obs_path,
            navigation_file=nav_path,
            station_name="AMC400USA",
            column_mapper={k: k for k in Epoch.OBSERVABLES},
        )
    )

    # Check the length of the epochified data
    assert len(epochified) == 120

    # Assert the length of individual epochs is either 9, 10, or 11
    for epoch in epochified:
        assert len(epoch) in [8, 9, 10, 11]

    # Assert that the epoch has both obs and nav data
    for epoch in epochified:
        assert epoch.obs_data is not None
        assert epoch.nav_data is not None

    # Assert that the epoch observation data has no NaN values
    # on following columns: ['C1C', "C2W", "L1C", "L2W"]
    for epoch in epochified:
        assert epoch.obs_data[["C1C", "C2W", "L1C", "L2W"]].isna().sum().sum() == 0


def test_mode(obsfilepath, navfilepath):
    """Mode must be one of 'maxsv' or 'nearest'. Raised ValueError otherwise"""
    obs = Path(obsfilepath)
    nav = Path(navfilepath)

    with pytest.raises(ValueError):
        list(from_rinex_files(obs, nav, mode="invalid_mode"))


def test_epochify_different_modes(obsfilepath, navfilepath):
    """Test epochify function with different modes"""
    obs = Path(obsfilepath)
    nav = Path(navfilepath)

    # Test mode 'meansv'
    epochified_mean = list(from_rinex_files(obs, nav, mode="nearest"))
    assert len(epochified_mean) == 120

    # Test mode 'maxsv'
    epochified_max = list(from_rinex_files(obs, nav, mode="maxsv"))
    assert len(epochified_max) == 120
