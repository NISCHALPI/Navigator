import pytest
import tempfile
from pathlib import Path
import datetime
from navigator.download.idownload.sp3.ccdis_igs_sp3 import NasaCddisIgsSp3


def test_nasa_sp3_download():
    """
    Test NASA CDDIS download
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        downloder = NasaCddisIgsSp3()

        # Dates
        d1 = datetime.datetime(2020, 1, 1)
        d2 = datetime.datetime(2023, 1, 25)

        # Download the SP3 files
        downloder.download_from_datetime(time=d1, save_dir=tmpdir)
        downloder.download_from_datetime(time=d2, save_dir=tmpdir)
        # Check that two files were downloaded
        assert len(list(Path(tmpdir).glob("*"))) == 2

        # Check that one was legacy and one new format
        assert any([downloder.matcher.match(f.name) for f in Path(tmpdir).glob("*")])
        assert any(
            [downloder.legacy_matcher.match(f.name) for f in Path(tmpdir).glob("*")]
        )
