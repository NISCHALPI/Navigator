import os

import pytest
from navigator.parse import IParseGPSNav, IParseGPSObs, Parser

__all__ = [
    "navfilepath",
    "obsfilepath",
    "nav_data",
    "obs_data",
    "sp3_legacy_filepath",
    "sp3_filepath",
]


@pytest.fixture
def navfilepath():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd, "rinexsamples/rnx/AMC400USA_R_20230391700_01H_GN.rnx")


@pytest.fixture
def obsfilepath():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd, "rinexsamples/rnx/AMC400USA_R_20230391700_01H_30S_MO.crx")


@pytest.fixture
def sp3_legacy_filepath():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd, "rinexsamples/rnx/igs19663.sp3.Z")


@pytest.fixture
def sp3_filepath():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(
        cwd, "rinexsamples/rnx/IGS0OPSFIN_20233090000_01D_15M_ORB.SP3.gz"
    )


@pytest.fixture
def skydel_sim_nav_filepath():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd, "rinexsamples/sim/SkydelRINEX_S_20242100_7200S_GN.rnx")


@pytest.fixture
def skydel_sim_g10_l1c():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd, "rinexsamples/sim/Skydel_G10_C1C.csv")


@pytest.fixture
def skydel_sim_obs_filepath():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd, "rinexsamples/sim/SkydelRINEX_S_20242100_7200S_MO.rnx")


@pytest.fixture
def skyder_true_reciever_state_filepath():
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd, "rinexsamples/sim/Skydel_reciever_pos.csv")


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
