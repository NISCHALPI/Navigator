"""This module provides a way to download RINEX files from the Australian Government GNSS data API portal.

The API documentation can be found at: https://data.gnss.ga.gov.au/docs/rinex-file-query/v1.0/web-api-access.html.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import requests

from ..idownload import IDownload

__all__ = ["AusGovDownload"]


class AusGovDownload(IDownload):
    """Implements an API download from the Australian Government GNSS data API portal.

    This class provides a way to download RINEX files from the Australian Government GNSS data API portal.
    The API documentation can be found at: https://data.gnss.ga.gov.au/docs/rinex-file-query/v1.0/web-api-access.html.

    Usage:
    To use this class, simply create an instance of it and call the `download` method with the desired parameters.
    """

    def __init__(self, max_workers: int = 1) -> None:
        """Initialize the IAusGovDownload instance.

        Args:
            max_workers (int, optional): The maximum number of worker threads to use for downloading. Defaults to 1.
        """
        assert max_workers > 0, "max_workers must be greater than 0"
        self._max_workers = max_workers
        self.API_URL = "https://data.gnss.ga.gov.au/api/rinexFiles"

        self.IGR0 = [
            "ALBY",
            "ALIC",
            "ANDA",
            "BAKO",
            "BLUF",
            "BMAN",
            "BNDY",
            "CEDU",
            "COOB",
            "DARW",
            "DUND",
            "GUAM",
            "HAAS",
            "HIL1",
            "HOB2",
            "HOKI",
            "JAB2",
            "KARR",
            "KAT1",
            "KGIS",
            "LAMB",
            "LAUT",
            "LORD",
            "MAC1",
            "MAJU",
            "MCHL",
            "MOBS",
            "MQZG",
            "MRO1",
            "MULG",
            "NAUR",
            "NIUM",
            "NIUT",
            "NLSN",
            "NORF",
            "NORS",
            "NTUS",
            "PERT",
            "PNGM",
            "POHN",
            "PTSV",
            "PTVL",
            "PYGR",
            "SA45",
            "SOLO",
            "STR1",
            "SYDN",
            "SYM1",
            "TAUP",
            "TBOB",
            "THEV",
            "TITG",
            "TOW2",
            "TUVA",
            "VGMT",
            "WARA",
            "WEIP",
            "WEST",
            "WGTN",
            "WHKT",
            "WILU",
            "WMGA",
            "XMIS",
            "YAR3",
            "YARR",
            "YEEL",
            "00NA",
            "ABMF",
            "ABNY",
            "AGGO",
            "AIRA",
            "AJAC",
            "ALBH",
            "ALGO",
            "ALRT",
            "AMC2",
            "ANKR",
            "ARTU",
            "BADG",
            "BAKE",
            "BALI",
            "BARH",
            "BBDH",
            "BERM",
            "BHR3",
            "BHR4",
            "BJCO",
            "BJFS",
            "BJNM",
            "BOGT",
            "BOR1",
            "BRAZ",
            "CLUM",
            "CHID",
            "CDWL",
        ]
        super().__init__(features="Rinex Downloader")

    def api_url(
        self,
        stationId: List[str],
        startTime: str,
        endTime: str,
        filePeriod: str = "01D",
        fileType: str = "obs",
        rinexVersion: str = "3",
        decompress: bool = False,  # noqa : ARG
        metadataStatus: str = "valid",  # noqa : ARG
    ) -> str:
        """Generate the API URL and query parameters.

        Args:
            stationId (List[str]): List of station IDs to query.
            startTime (datetime): Start date and time for the query. Should be in ISO 8601 format UTC.
            endTime (datetime): End date and time for the query. Should be in ISO 8601 format UTC.
            filePeriod (str, optional): RINEX file period. Defaults to '01D'. Can be '01D', '01H', '015M'.
            fileType (str, optional): RINEX file type. Defaults to 'obs'. Can b3 met, nav, obs.
            rinexVersion (str, optional): RINEX version. Defaults to '3'. Can be 2 or 3.
            decompress (bool, optional): Whether to decompress RINEX files. Defaults to False.
            metadataStatus (str, optional): Metadata validation status. Defaults to 'valid'. Can be 'valid', 'all', 'invalid'.

        Returns:
            str: The generated API URL.
        """
        # Check of startTimes and endTimes are in ISO 8601 format
        if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", startTime):
            raise ValueError("startTime should be in ISO 8601 format")

        # Convert stationId to list if it is a string
        if isinstance(stationId, str):
            stationId = [stationId]

        query_params = {
            "stationId": ",".join(stationId),
            "startDate": startTime,
            "endDate": endTime,
            "filePeriod": filePeriod,
            "fileType": fileType,
            "rinexVersion": rinexVersion,
        }

        # Manually create the URL
        return (
            self.API_URL
            + "?"
            + "&".join([f"{key}={value}" for key, value in query_params.items()])
        )

    def _download(
        self,
        stationId: str,
        startDateTime: datetime,
        endDateTime: datetime,
        filePeriod: str = "daily",
        fileType: str = "obs",
        rinexVersion: str = "3",
        decompress: bool = False,
        metadataStatus: str = "valid",
    ) -> List[dict]:
        """Perform the API request to download RINEX files.

        Args:
            stationId (str): The station ID to query.
            startDateTime (datetime): Start date and time for the query.
            endDateTime (datetime): End date and time for the query.
            filePeriod (str, optional): RINEX file period.Defaults to '01D'. Can be '01D', '01H', '015M'.
            fileType (str, optional): RINEX file type. Defaults to 'obs'.
            rinexVersion (str, optional): RINEX version. Defaults to '3'.
            decompress (bool, optional): Whether to decompress RINEX files. Defaults to False.
            metadataStatus (str, optional): Metadata validation status. Defaults to 'valid'.

        Returns:
            List[dict]: A list of downloaded RINEX file information.
        """
        # Convert datetime to ISO 8601 format
        startTime = startDateTime.strftime("%Y-%m-%dT%H:%M:%SZ")
        endTime = endDateTime.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Raise error if stationId is not in IGR0
        if stationId not in self.IGR0:
            raise ValueError(f"stationId must be in {self.IGR0}")

        # Get the API URL
        url = self.api_url(
            stationId,
            startTime,
            endTime,
            filePeriod,
            fileType,
            rinexVersion,
            decompress,
            metadataStatus,
        )

        # Request the data
        request = requests.get(url)
        try:
            request.raise_for_status()
        except requests.exceptions.HTTPError:
            # If the request fails, return a dummy dictionary
            return [
                {
                    'siteId': stationId,
                    'fileType': fileType,
                    'filePeriod': filePeriod,
                    'startDate': startTime,
                    'rinexVersion': rinexVersion,
                    'fileLocation': 'unavailable',
                    'metadataStatus': metadataStatus,
                    'createdAt': startTime,
                    'modifiedAt': startTime,
                    'fileId': 'unavailable',
                }
            ]

        return json.loads(request.content)

    def download(
        self,
        stationId: list[str],
        startDateTime: datetime,
        endDateTime: datetime,
        filePeriod: str = "01D",
        fileType: str = "obs",
        rinexVersion: str = "3",
        decompress: bool = False,
        metadataStatus: str = "valid",
        save: bool = False,
        save_path: Path = Path.cwd(),
    ) -> List[dict]:
        """Download the links to RINEX files.

        Args:
            stationId (List[str]): List of station IDs to query.
            startDateTime (datetime): Start date and time for the query.
            endDateTime (datetime): End date and time for the query.
            filePeriod (str, optional): RINEX file period. Defaults to '01D'. Can be '01D', '01H', '015M'.
            fileType (str, optional): RINEX file type. Defaults to 'obs'.
            rinexVersion (str, optional): RINEX version. Defaults to '3'.
            decompress (bool, optional): Whether to decompress RINEX files. Defaults to False.
            metadataStatus (str, optional): Metadata validation status. Defaults to 'valid'.
            save (bool, optional): Whether to save the downloaded files. Defaults to False.

        Returns:
            List[dict]: A list of downloaded RINEX file information.
        """
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                executor.submit(
                    self._download,
                    stationId,
                    startDateTime,
                    endDateTime,
                    filePeriod,
                    fileType,
                    rinexVersion,
                    decompress,
                    metadataStatus,
                )
                for stationId in stationId
            ]

            results = []
            for future in futures:
                # Add the results to the list
                results.extend(future.result())

            # Save the files if save is True
            if save:
                executor.map(self._save, results, [save_path] * len(results))

            return results

    def _save(self, file: dict, save_dir: Path) -> None:
        """Save the RINEX file to disk.

        Args:
            file (dict): The RINEX file information.
            save_dir (Path): The directory to save the file to.
        """
        # Check if the file is available
        if file["fileLocation"] == "unavailable":
            return

        # Parse the url to get the filename
        parsed_url = urlparse(file["fileLocation"])
        filename = Path(parsed_url.path).name

        # Get the content of the file
        r = requests.get(file["fileLocation"], allow_redirects=True)

        # Save the file
        with open(save_dir / filename, "wb") as f:
            f.write(r.content)

        return
