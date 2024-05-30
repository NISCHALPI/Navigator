"""The `IParseYumaAlm` class is an interface class for parsing Yuma Almanac files.

This module defines the `IParseYumaAlm` class, which is used to parse Yuma Almanac files and extract metadata and data into pandas Series and DataFrame respectively.

Attributes:
    pattern (re.Pattern): Regular expression pattern to match Yuma Almanac file format.

"""

import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from pandas.core.api import DataFrame as DataFrame
from pandas.core.api import Series as Series

from ..base_iparse import IParse

__all__ = ["IParseYumaAlm"]


class IParseYumaAlm(IParse):
    """The `IParseYumaAlm` class is an interface class for parsing Yuma Almanac files.

    This class provides methods to parse Yuma Almanac files and extract metadata and data into pandas Series and DataFrame respectively.

    """

    def __init__(self) -> None:
        """Initializes the `IParseYumaAlm` class."""
        self.pattern = re.compile(
            r"""
            \*{8}\sWeek\s(\d+)\salmanac\sfor\sPRN-(\d+)\s\*{8}\n
            ((?:\s*[^\n:]+:\s*[\d.E+-]+\s*\n)+)
        """,
            re.MULTILINE | re.VERBOSE,
        )
        super().__init__(features="Yuma_Almanac")

    def parse(
        self, filename: Path, **kwargs  # noqa: ARG002
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Parses the given Yuma Almanac file and returns a pandas DataFrame.

        Args:
            filename (Path): The filename to parse.
            **kwargs: Additional keyword arguments to pass to the parser.

        Returns:
            Tuple[pd.Series, pd.DataFrame]: A tuple containing the metadata and the data.
        """
        matches = self.pattern.findall(filename.read_text())
        return_dict = {}
        for _, prn, dta in matches:
            dta = dta.strip().replace(" ", "").replace("E", "e").split("\n")
            dta = dict([x.split(":") for x in dta])
            dta = {
                key: int(value) if key in ["ID", "Health", "week"] else float(value)
                for key, value in dta.items()
            }
            return_dict[f"G{prn}"] = dta

        return pd.Series(), pd.DataFrame.from_dict(return_dict, orient="index")
