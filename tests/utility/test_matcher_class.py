import pytest
from navigator.utility.matcher.matcher import (
    MixedObs3DailyMatcher,
    GpsNav3DailyMatcher,
    EpochFileMatcher,
)

mixed_obs_filename = [
    "BOGI00POL_R_20210140000_01D_30S_MO.crx.gz",
    "KOUR00GUF_R_20210120000_01D_30S_MO.crx.gz",
    "METG00FIN_R_20210120000_01D_30S_MO.crx.gz",
    "OWMG00NZL_R_20210140000_01D_30S_MO.crx.gz",
    "ROAG00ESP_R_20210100000_01D_30S_MO.crx.gz",
    "SUTH00ZAF_R_20210100000_01D_30S_MO.crx.gz",
]

mixed_nav_filename = [
    "BOGI00POL_R_20210140000_01D_GN.rnx.gz",
    "KOUR00GUF_R_20210120000_01D_GN.rnx.gz",
    "METG00FIN_R_20210120000_01D_GN.rnx.gz",
    "OWMG00NZL_R_20210140000_01D_GN.rnx.gz",
    "ROAG00ESP_R_20210100000_01D_GN.rnx.gz",
    "SUTH00ZAF_R_20210100000_01D_GN.rnx.gz",
]

epoch_filename = [
    "EPOCH_KOUR00GUF_202101122126.pkl",
    "EPOCH_KOUR00GUF_202101122127.pkl",
    "EPOCH_KOUR00GUF_202101122128.pkl",
    "EPOCH_KOUR00GUF_202101122129.pkl",
    "EPOCH_KOUR00GUF_202101122130.pkl",
    "EPOCH_KOUR00GUF_202101122131.pkl",
    "EPOCH_KOUR00GUF_202101122132.pkl",
    "EPOCH_KOUR00GUF_202101122133.pkl",
    "EPOCH_KOUR00GUF_202101122134.pkl",
    "EPOCH_KOUR00GUF_202101122135.pkl",
    "EPOCH_KOUR00GUF_202101122136.pkl",
]


def test_mixed_obs_matcher():
    mixed_obs_matcher = MixedObs3DailyMatcher()

    for filename in mixed_obs_filename:
        assert mixed_obs_matcher.match(filename)

    # Test Metadata for BOGI00POL_R_20210140000_01D_30S_MO.crx.gz
    metadata = mixed_obs_matcher.extract_metadata(mixed_obs_filename[0])

    assert metadata["marker_name"] == "BOGI"
    assert metadata["marker_number"] == "0"
    assert metadata["receiver_number"] == "0"
    assert metadata["country_code"] == "POL"
    assert metadata["data_type"] == "R"
    assert metadata["year"] == "2021"
    assert metadata["day_of_year"] == "014"
    assert metadata["hour"] == "00"
    assert metadata["minute"] == "00"
    assert metadata["file_extension"] == "crx"


def test_gps_nav_matcher():
    gps_nav_matcher = GpsNav3DailyMatcher()

    for filename in mixed_nav_filename:
        assert gps_nav_matcher.match(filename)

    # Test Metadata for BOGI00POL_R_20210140000_01D_GN.rnx.gz
    metadata = gps_nav_matcher.extract_metadata(mixed_nav_filename[0])

    assert metadata["marker_name"] == "BOGI"
    assert metadata["marker_number"] == "0"
    assert metadata["receiver_number"] == "0"
    assert metadata["country_code"] == "POL"
    assert metadata["data_type"] == "R"
    assert metadata["year"] == "2021"
    assert metadata["day_of_year"] == "014"
    assert metadata["hour"] == "00"
    assert metadata["minute"] == "00"
    assert metadata["file_extension"] == "rnx"


def test_epoch_matcher():
    epoch_matcher = EpochFileMatcher()

    for filename in epoch_filename:
        assert epoch_matcher.match(filename)

    # Test Metadata for EPOCH_KOUR00GUF_202101122126.pkl
    metadata = epoch_matcher.extract_metadata(epoch_filename[0])

    assert metadata["station_name"] == "KOUR00GUF"
    assert metadata["year"] == "2021"
    assert metadata["month"] == "01"
    assert metadata["day"] == "12"
    assert metadata["hour"] == "21"
    assert metadata["minute"] == "26"
