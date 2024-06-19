"""Matcher for the fragment."""

from .matcher import Matcher

__all__ = ["FragObsMatcher", "FragNavMatcher"]


class FragObsMatcher(Matcher):
    """Matcher for the fragment.

    Inherits from Matcher to handle pattern matching for observation fragments.

    Attributes:
        frag_regex (str): Regular expression for observation fragments.

    Methods:
        __init__(): Initialize the Matcher for the fragment.
        extract_metadata(filename: str) -> dict: Extract metadata from the filename.
    """

    frag_regex = (
        r"^OBSFRAG_([A-Z0-9]{9})_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.pkl$"
    )

    def __init__(self) -> None:
        """Initialize the Matcher for the fragment."""
        super().__init__(self.frag_regex)

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
                "second": match.group(7),
            }
        return {}


class FragNavMatcher(Matcher):
    """Matcher for the fragment.

    Inherits from Matcher to handle pattern matching for navigation fragments.

    Attributes:
        nav_regex (str): Regular expression for navigation fragments.

    Methods:
        __init__(): Initialize the Matcher for the fragment.
        extract_metadata(filename: str) -> dict: Extract metadata from the filename.
    """

    nav_regex = (
        r"^NAVFRAG_([A-Z0-9]{9})_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.pkl$"
    )

    def __init__(self) -> None:
        """Initialize the Matcher for the fragment."""
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
                "station_name": match.group(1),
                "year": match.group(2),
                "month": match.group(3),
                "day": match.group(4),
                "hour": match.group(5),
                "minute": match.group(6),
                "second": match.group(7),
            }
        return {}
