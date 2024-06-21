from navigator.epoch import from_rinex_files, Epoch
from pathlib import Path
from tests.common_fixtures import (
    skydel_sim_obs_filepath,
    skydel_sim_nav_filepath,
    skyder_true_reciever_state_filepath,
)
from navigator.core import Triangulate, IterativeTriangulationInterface
import pandas as pd
import numpy as np
import pytest


def test_gps_triangulation_on_sim_dataset(
    skydel_sim_obs_filepath,
    skydel_sim_nav_filepath,
    skyder_true_reciever_state_filepath,
):
    """Test GPS triangulation on simulated dataset.

    Note:
        The tropospheric and ionospheric models are not checked in this test.

    Args:
        skydel_sim_obs_filepath (str): Path to the simulated observation file.
        skydel_sim_nav_filepath (str): Path to the simulated navigation file.
        skyder_true_reciever_state_filepath (str): Path to the true receiver state file.
    """
    # Spacing
    N = 10
    # Create a column mapper
    column_mapper = {
        Epoch.L1_CODE_ON: "CA",
        Epoch.L2_CODE_ON: "CC",
        Epoch.L1_PHASE_ON: "LA",
        Epoch.L2_PHASE_ON: "LC",
    }
    # Create test epochs
    epochs = list(
        from_rinex_files(
            skydel_sim_obs_filepath,
            skydel_sim_nav_filepath,
            column_mapper=column_mapper,
        )
    )
    # Read the true receiver state
    true_state = pd.read_csv(skyder_true_reciever_state_filepath, index_col=0)
    position_colnames = ['ECEF X (m)', 'ECEF Y (m)', 'ECEF Z (m)']
    triangulate_pos_colnames = ['x', 'y', 'z']

    # Create Triangulate object
    triangulator = Triangulate(
        interface=IterativeTriangulationInterface(code_only=True)
    )
    # Triangulate on N spaced epochs
    df = triangulator.triangulate_time_series(epoches=epochs[::10])

    # Check that the norm of the difference between the true and estimated positions is less than 1 meter
    error = np.linalg.norm(
        df[triangulate_pos_colnames].values - true_state[position_colnames].values,
        axis=1,
    )

    # Check that the error is less than 1cm
    assert np.all(error < 0.01)

    # Check that the clock bias is approximately zero
    assert pytest.approx(df["cdt"].mean(), abs=0.01) == 0

    # Assert the UREA is less than 0.001m
    assert pytest.approx(df["UREA"].mean(), abs=0.001) == 0
