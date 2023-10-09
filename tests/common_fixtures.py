import pytest
import os


from navigator.parse import Parser, IParseGPSNav, IParseGPSObs

__all__ = [
    'navfilepath',
    'obsfilepath',
    'nav_data',
    'obs_data',
]


@pytest.fixture
def navfilepath():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd, 'rinexsamples/YELL00CAN_R_20231841500_01H_MN.rnx')


@pytest.fixture
def obsfilepath():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd, 'rinexsamples/YELL00CAN_R_20231841500_01H_30S_MO.crx')


@pytest.fixture
def nav_data(navfilepath) -> tuple:
    parser = Parser(
        iparser=IParseGPSNav(),
    )

    metadata, parsed_data = parser(filepath=navfilepath)

    return metadata, parsed_data


@pytest.fixture
def obs_data(obsfilepath) -> tuple:
    parser = Parser(
        iparser=IParseGPSObs(),
    )

    metadata, parsed_data = parser(filepath=obsfilepath)

    return metadata, parsed_data
