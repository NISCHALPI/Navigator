"""Matcher module.

This module has the Matcher classes that are used to match the files and the metadata based on the filename. Regex is used to match the filename.

Base Class Matcher is extended by all the other Matcher classes to extend the functionality.

Usage:
    # Example usage:
    mixed_obs_matcher = MixedObs3DailyMatcher()
    nav_matcher = Nav3DailyMatcher()

    file_to_check = 'BOGI00POL_R_20210140000_01D_30S_MO.crx.gz'

    if mixed_obs_matcher.match(file_to_check):
        metadata = mixed_obs_matcher.extract_metadata(file_to_check)
        print("File matched! Extracted metadata:", metadata)
    else:
        print("File does not match the pattern.")

"""

import re
from abc import ABC

__all__ = [
    "Matcher",
    "MixedObs3DailyMatcher",
    "GpsNav3DailyMatcher",
    "EpochFileMatcher",
]


class Matcher(ABC):
    """Base Matcher Class extended by all other Matcher classes.

    Args:
        ABC (_type_): Abstract Base Class.
    """

    def __init__(self, pattern: str) -> None:
        """Initialize the Matcher with a regex pattern.

        Args:
            pattern (str): Regular expression pattern to match the filename.
        """
        self.pattern = re.compile(pattern)

    def match(self, filename: str) -> bool:
        """Match the filename with the regex pattern.

        Args:
            filename (str): Filename to match.

        Returns:
            bool: True if the filename matches the regex pattern, False otherwise.
        """
        match = self.pattern.match(filename)
        return bool(match)

    def __call__(self, filename: str) -> bool:
        """Call the match method.

        Args:
            filename (str): Filename to match.

        Returns:
            bool: True if the filename matches the regex pattern, False otherwise.
        """
        return self.match(filename)


class MixedObs3DailyMatcher(Matcher):
    """Matcher for RINEX v3 Daily Mixed OBS files based on CDDIS naming convention.

    Args:
        Matcher (_type_): Base Matcher Class.
    """

    obs_regex = r"^([A-Z0-9]{4})([0-9]{1})([0-9]{1})([A-Z]{3})_([RSU])_(\d{4})(\d{3})(\d{2})(\d{2})_01D_30S_MO\.(crx|rnx)\.gz"

    def __init__(self) -> None:
        """Initialize the Matcher for RINEX v3 Daily Mixed OBS files."""
        super().__init__(self.obs_regex)

    def extract_metadata(self, filename: str) -> dict:
        """Extract metadata from the filename.

        Args:
            filename (str): Filename to extract metadata from.

        Returns:
            dict: Extracted metadata from the filename.
        """
        match = self.pattern.match(filename)
        if match:
            return {
                "marker_name": match.group(1),
                "marker_number": match.group(2),
                "receiver_number": match.group(3),
                "country_code": match.group(4),
                "data_type": match.group(5),
                "year": match.group(6),
                "day_of_year": match.group(7),
                "hour": match.group(8),
                "minute": match.group(9),
                "file_extension": match.group(10),
                "station_name": f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}",
            }
        return {}


class GpsNav3DailyMatcher(Matcher):
    """Matcher for RINEX v3 Daily NAV files based on CDDIS naming convention.

    Args:
        Matcher (_type_): Base Matcher Class.
    """

    nav_regex = r"^([A-Z0-9]{4})([0-9]{1})([0-9]{1})([A-Z]{3})_([RSU])_(\d{4})(\d{3})(\d{2})(\d{2})_01D_GN\.(crx|rnx)\.gz"

    def __init__(self) -> None:
        """Initialize the Matcher for RINEX v3 Daily NAV files."""
        super().__init__(self.nav_regex)

    def extract_metadata(self, filename: str) -> dict:
        """Extract metadata from the filename.

        Args:
            filename (str): Filename to extract metadata from.

        Returns:
            dict: Extracted metadata from the filename.
        """
        match = self.pattern.match(filename)
        if match:
            return {
                "marker_name": match.group(1),
                "marker_number": match.group(2),
                "receiver_number": match.group(3),
                "country_code": match.group(4),
                "data_type": match.group(5),
                "year": match.group(6),
                "day_of_year": match.group(7),
                "hour": match.group(8),
                "minute": match.group(9),
                "file_extension": match.group(10),
                "station_name": f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}",
            }

        return {}


class EpochFileMatcher(Matcher):
    """Matcher for the epoch files generated by the navigator.utility.v3daily.data.epoch_directory.EpochDirectory class."""

    epoch_regex = (
        r"^EPOCH_([A-Z0-9]{9})_([0-9]{4})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})\.pkl$"
    )

    def __init__(self) -> None:
        """Initialize the Matcher for the epoch files."""
        super().__init__(self.epoch_regex)

    def extract_metadata(self, filename: str) -> dict:
        """Extract metadata from the filename.

        Args:
            filename (str): Filename to extract metadata from.

        Returns:
            dict: Extracted metadata from the filename.
        """
        match = self.pattern.match(filename)
        if match:
            return {
                "station_name": match.group(1),
                "year": match.group(2),
                "month": match.group(3),
                "day": match.group(4),
                "hour": match.group(5),
                "minute": match.group(6),
            }
        return {}
