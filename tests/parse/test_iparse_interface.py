import pytest
from  navigator.parse import Parser, IParseGPSNav, IParseGPSObs
import os
import pandas as pd



@pytest.fixture
def navfile():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd , 'rinexsamples/YELL00CAN_R_20231841500_01H_MN.rnx')


@pytest.fixture
def obsfile():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd , 'rinexsamples/YELL00CAN_R_20231841500_01H_30S_MO.crx')




def test_gps_observational_interface(obsfile) -> None:
    
    # Instantiate the parser
    parser = Parser(
        iparser=IParseGPSObs(),
    )
    
    metadata, parsed_data = parser(filepath=obsfile)

    
    assert isinstance(metadata, pd.Series)
    assert isinstance(parsed_data, pd.DataFrame)
    assert 'C1C' in parsed_data.columns
    assert 'L1C' in parsed_data.columns
    assert metadata['rinextype'] == 'obs'

    return

    
    
def test_gps_navigation_interface(navfile) -> None:
    
    # Instantiate the parser
    parser = Parser(
        iparser=IParseGPSNav(),
    )
    
    metadata, parsed_data = parser(filepath=navfile)
    
    assert isinstance(metadata, pd.Series)
    assert isinstance(parsed_data, pd.DataFrame)
    assert 'SVclockBias' in parsed_data.columns
    assert 'SVclockDrift' in parsed_data.columns
    assert metadata['rinextype'] == 'nav'
    
    return

