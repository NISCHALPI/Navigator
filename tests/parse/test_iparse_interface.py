import pytest
from navigator.parse import Parser, IParseGPSNav, IParseGPSObs
import os
import pandas as pd

from tests.common_fixtures import navfilepath, obsfilepath

def test_gps_observational_interface(obsfilepath) -> None:
    # Instantiate the parser
    parser = Parser(
        iparser=IParseGPSObs(),
    )

    metadata, parsed_data = parser(filepath=obsfilepath)

    assert isinstance(metadata, pd.Series)
    assert isinstance(parsed_data, pd.DataFrame)
    assert 'C1C' in parsed_data.columns
    assert 'L1C' in parsed_data.columns
    assert metadata['rinextype'] == 'obs'

    return


def test_gps_navigation_interface(navfilepath) -> None:
    # Instantiate the parser
    parser = Parser(
        iparser=IParseGPSNav(),
    )

    metadata, parsed_data = parser(filepath=navfilepath)

    assert isinstance(metadata, pd.Series)
    assert isinstance(parsed_data, pd.DataFrame)
    assert 'SVclockBias' in parsed_data.columns
    assert 'SVclockDrift' in parsed_data.columns
    assert metadata['rinextype'] == 'nav'

    return
