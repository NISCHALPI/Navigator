import pytest
from navigator.utility.matcher import SP3Matcher, LegacySP3Matcher


@pytest.fixture
def sp3_matcher():
    return SP3Matcher()


@pytest.fixture
def legacy_sp3_matcher():
    return LegacySP3Matcher()


def test_sp3_matcher_extract_metadata(sp3_matcher):
    filename = "IGS0OPSFIN_20233090000_01D_15M_ORB.SP3.gz"
    metadata = sp3_matcher.extract_metadata(filename)
    assert metadata == {
        "analysis_center": "IGS",
        "campaign": "OPS",
        "solution_type": "FIN",
        "year": "2023",
        "day": "309",
        "hour": "00",
        "minute": "00",
        "length": "01D",
        "sampling": "15M",
        "content": "ORB",
        "format": "SP3",
    }


def test_legacy_sp3_matcher_extract_metadata(legacy_sp3_matcher):
    filename = "igr00100.05i.Z"
    metadata = legacy_sp3_matcher.extract_metadata(filename)
    assert metadata == {
        "analysis_center": "igr",
        "gps_week": "0010",
        "day": "0",
        "format": "05i",
    }


def test_sp3_matcher_invert(sp3_matcher):
    inverted_filename = sp3_matcher.invert(2023, 309)
    assert inverted_filename == "IGS0OPSFIN_20233090000_01D_15M_ORB.SP3.gz"


def test_legacy_sp3_matcher_invert(legacy_sp3_matcher):
    inverted_filename = legacy_sp3_matcher.invert(2022, 1)
    assert inverted_filename == "igs20221.sp3.Z"


def test_sp3_matcher_invert_with_hour_minute(sp3_matcher):
    inverted_filename = sp3_matcher.invert(2023, 309, hour=12, minute=30)
    assert inverted_filename == "IGS0OPSFIN_20233091230_01D_15M_ORB.SP3.gz"


def test_metadata_errors(sp3_matcher):
    with pytest.raises(AssertionError):
        sp3_matcher.invert(20232, 302)

    with pytest.raises(AssertionError):
        sp3_matcher.invert(2023, 3092)

    with pytest.raises(AssertionError):
        sp3_matcher.invert(2023, 309, hour=242)
