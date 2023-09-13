"""The `abstract_parser` module provides the `AbstractParser` class, an abstract base class designed for parsing data from files using an IParse interface. This module serves as the foundation for implementing specific parsers for various file formats.

Module Contents:
    - AbstractParser: An abstract base class for parsing data from files.

Usage:
    To create a custom parser for a specific file format, inherit from the `AbstractParser` class and implement the required parsing logic by defining an IParse interface.

Example:
    ```python
    from myparser import MyCustomParser
    from myiparse import MyIParseImplementation

    # Create a parser instance
    custom_parser = MyCustomParser(MyIParseImplementation())

    # Parse data from a file
    parsed_data = custom_parser('datafile.dat')
    ```

Classes:
    - AbstractParser: An abstract base class for parsing data from files.

"""
import typing as tp
from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path

import pandas as pd

from .iparse import IParse  # TODO: add IParse

__all__ = ["AbstractParser", "Parser"]


class AbstractParser(ABC):
    """Abstract class for parsing data from a file using an IParse interface.

    This abstract class serves as a template for creating parsers for specific file formats. It enforces the use of an IParse interface for parsing data and provides common functionality for checking file integrity and handling logging.

    Attributes:
    iparser (IParse): An instance of an IParse interface for data parsing.
    logger (Logger | None): A logger for logging operations. Defaults to None.
    dispatcher (None): A parallel task dispatcher (optional). Defaults to None.

    Methods:
    __init__(self, iparser: IParse, logger: Logger | None = None, dispatcher: None = None) -> None:
    Initialize a new AbstractParser instance.

    _check_file_integrity(self, filepath: str | Path) -> None:
    Check the integrity of a file path to ensure it exists, is a file, and is readable.

    _parse(self, filepath: str | Path) -> tp.Tuple[pd.Series, pd.DataFrame]:
    Parse data from a file using the provided IParse interface.

    __call__(self, filepath: str | Path) -> Any:
    Call method for parsing data from a file.

    __repr__(self) -> str:
    Return a string representation of the AbstractParser instance.

    Example:
    ```python
    from myparser import MyCustomParser
    from myiparse import MyIParseImplementation

    # Create a parser instance
    custom_parser = MyCustomParser(MyIParseImplementation())

    # Parse data from a file
    parsed_data = custom_parser('datafile.dat')
    ```

    """

    def __init__(
        self, iparser: IParse, logger: Logger | None = None, dispatcher: None = None
    ) -> None:  # TODO: add dispatcher
        """Initialize a new AbstractParser instance.

        Args:
            iparser (IParse): An instance of an IParse interface for data parsing.
            logger (Logger | None, optional): A logger for logging operations. Defaults to None.
            dispatcher (None, optional): A parallel task dispatcher (optional). Defaults to None.

        Raises:
            TypeError: If iparser is not an instance of IParse.
            TypeError: If logger is provided but is not an instance of Logger.

        """
        # Verify iparser is IParse
        if not issubclass(iparser.__class__, IParse):
            raise TypeError(f"iparser must be subclass IParse, not {type(iparser)}")
        self.iparser = iparser

        # Verify logger is Logger
        if logger is not None and not isinstance(logger, Logger):
            raise TypeError(f"logger must be Logger, not {type(logger)}")
        self.logger = logger

        # TO DO: Verify dispatcher is Dispatcher
        self.dispatcher = dispatcher

        return

    def _check_file_integrity(self, filepath: str | Path) -> None:
        """Check the integrity of a file path.

        Args:
            filepath (str | Path): The path to the file to check.

        Raises:
            TypeError: If filepath is not a str or Path.
            FileNotFoundError: If the file does not exist.
            FileNotFoundError: If the path is not a file.
            PermissionError: If the file is not readable.

        """
        # Verify filepath
        if not isinstance(filepath, (str, Path)):
            raise TypeError(f"filepath must be str or Path, not {type(filepath)}")

        if isinstance(filepath, str):
            filepath = Path(filepath)
        else:
            filepath = filepath

        # If not exists, raise error
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist")

        # If not a file, raise error
        if not filepath.is_file():
            raise FileNotFoundError(f"{filepath} is not a file")

        return

    # TO DO : def _dispatch(self) -> None:

    @abstractmethod
    def _parse(self, filepath: str | Path) -> tp.Tuple[pd.Series, pd.DataFrame]:
        """Parse data from a file using the provided IParse interface.

        Args:
            filepath (str | Path): Path to the file to parse.

        Returns:
            tp.Tuple[pd.Series, pd.DataFrame]: Tuple of parsed data (metadata, data).

        """
        # Verify filepath
        self._check_file_integrity(filepath)

        # Parse data
        return self.iparser(filepath)

    def __call__(self, filepath: str | Path) -> tp.Tuple[pd.Series, pd.DataFrame]:
        """Call method for parsing data from a file.

        Args:
            filepath (str | Path): Path to the file to parse.

        Returns:
            Any: The parsed data.

        """
        return self._parse(filepath)

    def __repr__(self) -> str:
        """Return a string representation of the AbstractParser instance.

        Returns:
            str: String representation of the object.

        """
        return f"{self.__class__.__name__}(iparser={self.iparser}, logger={self.logger}, dispatcher={self.dispatcher})"


class Parser(AbstractParser):
    """Concrete class for parsing data from a file using an IParse interface.

    Args:
        AbstractParser (_type_): Abstract class for parsing data from a file using an IParse interface. This is due to the builder design pattern.
    """

    def __init__(
        self, iparser: IParse, logger: Logger | None = None, dispatcher: None = None
    ) -> None:
        """Initialize a new AbstractParser instance.

        Args:
            iparser (IParse): An instance of an IParse interface for data parsing.
            logger (Logger | None, optional): A logger for logging operations. Defaults to None.
            dispatcher (None, optional): A parallel task dispatcher (optional). Defaults to None.

        Raises:
            TypeError: If iparser is not an instance of IParse.
            TypeError: If logger is provided but is not an instance of Logger.

        """
        super().__init__(iparser, logger, dispatcher)
        return

    def _parse(self, filepath: str | Path) -> tp.Tuple[pd.Series, pd.DataFrame]:
        return super()._parse(filepath)
