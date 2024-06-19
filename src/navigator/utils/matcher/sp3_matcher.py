"""Matcher for IGS SP3 files."""

from .matcher import Matcher

__all__ = ["SP3Matcher", "LegacySP3Matcher"]


class SP3Matcher(Matcher):
    """Matcher for SP3 files.

    This class is designed to match the new name format for SP3 files since GPS Week 2238.

    The SP3 file format is described at:
    - https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/orbit_products.html

    Attributes:
        sp3_pattern (str): Regular expression pattern for matching SP3 file names.
    """

    sp3_pattern = r"^([A-Z]{3})0([A-Z]{3})([A-Z]{3})_(\d{4})(\d{3})(\d{2})(\d{2})_([A-Z0-9]{3})_([A-Z0-9]{3})_([A-Z]{3})\.([A-Z0-9]{3})\.gz"

    def __init__(
        self,
    ) -> None:
        """Initialize the Matcher for SP3 files."""
        super().__init__(pattern=self.sp3_pattern)

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
                "analysis_center": match.group(1),
                "campaign": match.group(2),
                "solution_type": match.group(3),
                "year": match.group(4),
                "day": match.group(5),
                "hour": match.group(6),
                "minute": match.group(7),
                "length": match.group(8),
                "sampling": match.group(9),
                "content": match.group(10),
                "format": match.group(11),
            }
        return {}

    def invert(
        self,
        year: int,
        day: int,
        hour: int = 00,
        minute: int = 00,
        analysis_center: str = "IGS",
        campaign: str = "OPS",
        solution_type: str = "FIN",
        length: str = "01D",
        sampling: str = "15M",
        content: str = "ORB",
        format: str = "SP3",
    ) -> str:
        """Invert the metadata back to a filename.

        Args:
            year (int): The year of the metadata.
            day (int): The day of the metadata.
            hour (int): The hour of the metadata.
            minute (int): The minute of the metadata.
            analysis_center (str, optional): The analysis center. Defaults to "IGS".
            campaign (str, optional): The campaign. Defaults to "OPS".
            solution_type (str, optional): The solution type. Defaults to "FIN".
            length (str, optional): The length of the metadata. Defaults to "01D".
            sampling (str, optional): The sampling rate of the metadata. Defaults to "15M".
            content (str, optional): The content of the metadata. Defaults to "ORB".
            format (str, optional): The format of the metadata. Defaults to "SP3".

        Returns:
            str: The inverted filename.
        """
        # Assert that the metadata is valid
        assert len(str(year)) == 4, "Year must be 4 digits long"
        assert day <= 366, "Day must be between 0 and 365"
        assert hour <= 24, "Hour must be between 0 and 23"
        assert minute <= 60, "Minute must be between 0 and 59"
        assert len(analysis_center) == 3, "Analysis center must be 3 characters long"
        assert len(campaign) == 3, "Campaign must be 3 characters long"
        assert len(solution_type) == 3, "Solution type must be 3 characters long"
        assert len(length) == 3, "Length must be 3 characters long"
        assert len(sampling) == 3, "Sampling must be 3 characters long"
        assert len(content) == 3, "Content must be 3 characters long"
        assert len(format) == 3, "Format must be 3 characters long"

        return f"{analysis_center}0{campaign}{solution_type}_{str(int(year)).zfill(4)}{str(int(day)).zfill(3)}{str(int(hour)).zfill(2)}{str(int(minute)).zfill(2)}_{length}_{sampling}_{content}.{format}.gz"


class LegacySP3Matcher(Matcher):
    """Matcher for legacy SP3 files.

    This class is designed to match the legacy name format for SP3 files before GPS Week 2237.

    The SP3 file format is described at:
    - https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/orbit_products.html

    Attributes:
        sp3_pattern (str): Regular expression pattern for matching SP3 file names.
    """

    sp3_pattern = r"^([a-z]{3})(\d{4})([0-7]{1})\.([a-z0-9]{3})\.Z"

    def __init__(
        self,
    ) -> None:
        """Initialize the Matcher for SP3 files."""
        super().__init__(pattern=self.sp3_pattern)

    def extract_metadata(self, filename: str) -> dict | None:
        """Extract metadata from the filename.

        Args:
            filename (str): Filename to extract metadata from.

        Returns:
            dict: Extracted metadata from the filename.
        """
        match = self.pattern.match(filename)

        if match:
            return {
                "analysis_center": match.group(1),
                "gps_week": match.group(2),
                "day": match.group(3),
                "format": match.group(4),
            }
        return None

    def invert(
        self,
        gps_week: int,
        day: int,
        analysis_center: str = "igs",
        format: str = "sp3",
    ) -> str:
        """Invert the metadata back to a filename.

        Args:
            gps_week (int): _description_
            day (int): _description_
            analysis_center (str, optional): _description_. Defaults to "igs".
            format (str, optional): _description_. Defaults to "sp3".

        Returns:
            str: _description_
        """
        # Assert that the metadata is valid
        assert day in range(8), "Day must be between 0 and 7"
        assert len(analysis_center) == 3, "Analysis center must be 3 characters long"
        assert len(format) == 3, "Format must be 3 characters long"

        return f"{analysis_center}{str(gps_week).zfill(4)}{str(int(day)).zfill(1)}.{format}.Z"
