import pytest
from navigator.parse import Parser, IParseSP3GPS
import os
import pandas as pd

from tests.common_fixtures import sp3_filepath, sp3_legacy_filepath


def test_gps_sp3_legacy_parse(sp3_legacy_filepath):
    # Instantiate the parser
    parser = Parser(iparser=IParseSP3GPS())

    met, data = parser(filepath=sp3_legacy_filepath)

    assert isinstance(met, pd.Series), "Metadata is not a pandas Series"
    assert isinstance(data, pd.DataFrame), "Data is not a pandas DataFrame"

    assert "x" in data.columns, "x column not found"
    assert "y" in data.columns, "y column not found"
    assert "z" in data.columns, "z column not found"
    assert "clock" in data.columns, "clock_bias column not found"
    assert "dclock" in data.columns, "clock_drift column not found"


def test_gps_sp3_parse(sp3_filepath):
    # Instantiate the parser
    parser = Parser(iparser=IParseSP3GPS())

    met, data = parser(filepath=sp3_filepath)

    assert isinstance(met, pd.Series), "Metadata is not a pandas Series"
    assert isinstance(data, pd.DataFrame), "Data is not a pandas DataFrame"

    assert "x" in data.columns, "x column not found"
    assert "y" in data.columns, "y column not found"
    assert "z" in data.columns, "z column not found"
    assert "clock" in data.columns, "clock_bias column not found"
    assert "dclock" in data.columns, "clock_drift column not found"
