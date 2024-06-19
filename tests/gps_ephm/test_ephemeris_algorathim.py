import pytest
from navigator.parse import Parser, IParseGPSNav
from navigator.core import Satellite, IGPSEphemeris
from tests.common_fixtures import (
    safran_sim_nav_filepath,
    safran_sim_true_state_filepath,
)
import numpy as np
import pandas as pd


def test_ephemeris_alograthim(safran_sim_nav_filepath, safran_sim_true_state_filepath):
    # Unpack the nav_data
    metadata, data = IParseGPSNav().parse(filename=safran_sim_nav_filepath)
    # Read the true state csv file
    g10_true = pd.read_csv(safran_sim_true_state_filepath)

    # Create a gps ephemris processor
    ephemeris_processor = Satellite(iephemeris=IGPSEphemeris())
    state_cols = ['ECEF X (m)', 'ECEF Y (m)', 'ECEF Z (m)', 'Clock Correction (s)']

    # Start time of simulation
    startTime = pd.Timestamp.fromisoformat("2024-01-02 10:00:00")
    svTime = [
        startTime + pd.Timedelta(ms, unit="ms")
        for ms in g10_true["PSR satellite time (ms)"]
    ]

    state = []

    for t in svTime:
        state.append(
            ephemeris_processor(
                t=t,
                metadata=metadata,
                data=data.xs("G10", level="sv", drop_level=False),
            )
        )
    # Convert the state to a DataFrame
    state = pd.DataFrame(np.vstack(state), columns=state_cols)

    # Assert the state is equal to the true state
    assert np.allclose(state.values, g10_true[state_cols].values, atol=1e-5)
