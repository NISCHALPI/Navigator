"""Module for the IParse interface."""

import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

__all__ = ["IParse"]


class IParse(ABC):
    """Interface for parsing data from a file.

    This interface defines the structure for classes responsible for parsing data from various file formats.
    Implementing classes should provide a concrete implementation of the `_parse` method to handle the parsing logic.

    Attributes:
        _features (str): The type of data this parser is designed for (e.g., GPS, Glonass, Beidou, etc.).

    Methods:
        __init__(self, features: str | None) -> None:
            Initialize an IParse instance with the specified features type.

        __repr__(self) -> str:
            Return a string representation of the IParse instance, including its features type.

        parse(self, file: str) -> tp.Tuple[pd.Series, pd.DataFrame]:
            [Abstract Method] Parse data from a file and return a tuple containing metadata (as a pd.Series) and data (as a pd.DataFrame).

        __call__(self, filepath: Path) -> Any:
            Call method for parsing data from a file using the `_parse` method.

    Example:
        ```python
        class MyCustomParser(IParse):
            def __init__(self, features: str | None):
                super().__init__(features)

            def _parse(self, file: str) -> tp.Tuple[pd.Series, pd.DataFrame]:
                # Implement the parsing logic here
                # Return metadata and data as pd.Series and pd.DataFrame, respectively
                pass

        # Create an instance of the custom parser
        custom_parser = MyCustomParser("GPS")

        # Parse data from a file
        parsed_data = custom_parser(Path("datafile.dat"))
        ```

    """

    def __init__(self, features: str | None) -> None:
        """Initialize an IParse instance with the specified features type.

        Args:
            features (str | None): The type of data this parser is designed for (e.g., GPS, Glonass, Beidou, etc.).
                If not provided, the default value is "NoneType."

        """
        # Type of iparser (e.g. GPS, Glonass, Biedu etc.)
        self._features = features if features else "NoneType"
        super().__init__()

    def __repr__(self) -> str:
        """Return a string representation of the IParse instance, including its features type.

        Returns:
            str: A string representation of the object.

        """
        return f"Iparser({self._features})" if self._features else "Iparser()"

    @abstractmethod
    def parse(self, filename: str, **kwargs) -> tp.Tuple[pd.Series, pd.DataFrame]:
        """[Abstract Method] Parse data from a file.

        Args:
            filename (str): The path to the file to be parsed.
            kwargs: Additional keyword arguments to pass to the parser.

        Returns:
            tp.Tuple[pd.Series, pd.DataFrame]: A tuple containing parsed metadata (as a pd.Series) and data (as a pd.DataFrame).

        """
        pass

    def __call__(self, filepath: Path, **kwargs) -> tp.Tuple[pd.Series, pd.DataFrame]:
        """Call method for parsing data from a file using the `_parse` method.

        Args:
            filepath (Path): The path to the file to be parsed.
            kwargs: Additional keyword arguments to pass to the parser.

        Returns:
            Any: The result of the parsing operation, typically a tuple containing metadata and data.

        """
        return self.parse(filepath, **kwargs)
