import pytest
import tempfile
from pathlib import Path
import os
from navigator.download.idownload.rinex.nasa_cddis import NasaCddisV3


@pytest.mark.skipif(
    os.environ.get("CONNECT", None) == None,
    reason="Run this test only when explicitly network is enabled",
)
def test_nasa_download():
    """
    Test NASA CDDIS download
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        downloder = NasaCddisV3()

        # Download two rinex files
        downloder.download(
            year=2022,
            day=1,
            save_path=Path(tmpdir),
            num_files=1,
            no_pbar=True,
        )

        # Check that two files were downloaded
        assert len(list(Path(tmpdir).glob("*.gz"))) == 2
