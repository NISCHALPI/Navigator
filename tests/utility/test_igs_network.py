import pytest
import pandas as pd
from navigator.utility.igs import IGSNetwork
import numpy as np

# Fixture to create an instance of IGSNetwork for testing
@pytest.fixture
def igs_network():
    return IGSNetwork()

# Test to check if the IGSNetwork object is initialized properly
def test_igs_network_initialization(igs_network):
    assert isinstance(igs_network.stations, pd.DataFrame)
    assert len(igs_network) > 0

# Test to check if a specific station is present in the IGS network
def test_station_presence(igs_network):
    assert "ABMF00GLP" in igs_network
    assert "INVALID_STATION" not in igs_network

# Test to get the XYZ coordinates of a station
def test_get_xyz(igs_network):
    xyz_coords = igs_network.get_xyz("ABMF00GLP")
    assert isinstance(xyz_coords, np.ndarray)
    assert xyz_coords.shape == (3,)
    assert all(isinstance(coord, np.float64) for coord in xyz_coords)

# Test to get the ellipsoid details of a station
def test_get_ellipsoid(igs_network):
    ellipsoid_details = igs_network.get_ellipsoid("ABMF00GLP")
    assert isinstance(ellipsoid_details, np.ndarray)
    assert ellipsoid_details.shape == (3,)
    # Check if the ellipsoid details are floats
    assert all(isinstance(coord, np.float64) for coord in ellipsoid_details)

# Test to retrieve detailed information for a specific station
def test_get_igs_station(igs_network):
    station_info = igs_network.get_igs_station("ABMF00GLP")
    assert isinstance(station_info, pd.Series)
    assert "Latitude" in station_info.index
    assert "Longitude" in station_info.index
    assert "Height" in station_info.index

# Test error calculation for a specific station and measurement
def test_error_calculation(igs_network):
    error = igs_network.error("ABMF00GLP", 1000.0, 2000.0, 3000.0)
    assert isinstance(error, float)

# Test string representation of the IGSNetwork object
def test_string_representation(igs_network):
    assert str(igs_network) == "IGSNetwork(stations=514)"

# Test iteration through IGS station names
def test_iteration_through_stations(igs_network):
    for station in igs_network:
        assert isinstance(station, str)

# Test accessing a specific station's information using indexing
def test_access_station_info_with_indexing(igs_network):
    station_info = igs_network["ABMF00GLP"]
    assert isinstance(station_info, pd.Series)
    assert "Latitude" in station_info.index
    assert "Longitude" in station_info.index
    assert "Height" in station_info.index
