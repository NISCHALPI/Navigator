import pandas as pd
import pytest
from navigator.utility.epoch import Epoch
from tests.common_fixtures import nav_data, obs_data, navfilepath, obsfilepath

def test_epochify(obsfilepath, navfilepath, nav_data, obs_data):
    """Test the epochify function"""

    # Get the parsed nav and obs data from fixtures
    meta_nav, nav = nav_data
    meta_obs, obs = obs_data

    # Epochify the data
    epochified = Epoch.epochify(obs=obs, nav=nav, mode='maxsv')

    # Check the length of the epochified data
    assert len(epochified) == 120

    # Assert the length of individual epochs is either 9, 10, or 11
    for epoch in epochified:
        assert len(epoch) in [8, 9, 10, 11]

    # Assert that the epoch has both obs and nav data
    for epoch in epochified:
        assert epoch.obs_data is not None
        assert epoch.nav_data is not None
    



def test_epochify_input_type():
    """Test epochify function with invalid input type"""
    invalid_obs = "invalid_obs_data"
    invalid_nav = "invalid_nav_data"
    
    with pytest.raises(TypeError):
        Epoch.epochify(obs=invalid_obs, nav=nav_data[1], mode='maxsv')
        
    with pytest.raises(TypeError):
        Epoch.epochify(obs=obs_data[1], nav=invalid_nav, mode='maxsv')
        
    with pytest.raises(TypeError):
        Epoch.epochify(obs=invalid_obs, nav=invalid_nav, mode='maxsv')


def test_mode(obsfilepath, navfilepath, nav_data, obs_data):
    """Mode must be one of 'maxsv' or 'nearest'. Raised ValueError otherwise"""
    _, nav = nav_data
    _, obs = obs_data
    
    with pytest.raises(ValueError):
        Epoch.epochify(obs=obs, nav=nav, mode='invalid_mode')
    


def test_epochify_different_modes(obsfilepath, navfilepath, nav_data, obs_data):
    """Test epochify function with different modes"""
    _, nav = nav_data
    _, obs = obs_data

    # Test mode 'meansv'
    epochified_mean = Epoch.epochify(obs=obs, nav=nav, mode='nearest')
    assert len(epochified_mean) == 120
    
    # Test mode 'maxsv'
    epochified_max = Epoch.epochify(obs=obs, nav=nav, mode='maxsv')
    assert len(epochified_max) == 120
    
