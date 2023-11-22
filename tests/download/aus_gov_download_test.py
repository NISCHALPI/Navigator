import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock
import os

from navigator.download.idownload.rinex import AusGovDownload


@pytest.fixture
def aus_gov_downloader():
    return AusGovDownload(max_workers=2)


def test_api_url(aus_gov_downloader):
    # Define test parameters
    station_ids = ["ALBY", "ALIC"]
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    end_time = datetime(2023, 1, 2, 0, 0, 0)

    # Generate the API URL
    api_url = aus_gov_downloader.api_url(
        stationId=station_ids,
        startTime=start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        endTime=end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    # Define the expected URL based on the provided parameters
    expected_url = (
        "https://data.gnss.ga.gov.au/api/rinexFiles?"
        "stationId=ALBY,ALIC"
        "&startDate=2023-01-01T00:00:00Z"
        "&endDate=2023-01-02T00:00:00Z"
        "&filePeriod=01D"
        "&fileType=obs"
        "&rinexVersion=3"
    )

    # Assert that the generated API URL matches the expected URL
    assert api_url == expected_url


@pytest.mark.skipif(
    "RUN_DOWNLOAD_TESTS" in os.environ,
    reason="Run this test only when explicitly requested.",
)
def test_download(aus_gov_downloader, tmp_path):
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 2)
    result = aus_gov_downloader.download(
        stationId=["ALBY", "ALIC"],
        startDateTime=start_date,
        endDateTime=end_date,
        filePeriod="01D",
        fileType="obs",
        rinexVersion="3",
        decompress=False,
        metadataStatus="valid",
        save=True,
        save_path=tmp_path,
    )
    assert len(result) == 4  # Two stations are queried
    for file in result:
        assert file["fileLocation"] != "unavailable"


def test_save(aus_gov_downloader, tmp_path):
    file_info = {
        "siteId": "ALBY",
        "fileLocation": "http://example.com/rinex/ALBY_obs_20230101_00D_30S_MO.crx",
    }
    save_dir = tmp_path / "saved_files"
    save_dir.mkdir()
    aus_gov_downloader._save(file_info, save_dir)
    saved_files = list(save_dir.iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].name == "ALBY_obs_20230101_00D_30S_MO.crx"
