"""The `IParseSP3GPS` module provides an interface class for parsing SP3 (Standard Precise Orbit Ephemeris) files related to GPS data.

This module contains the `IParseSP3GPS` class, which is designed to parse SP3 files and extract GPS position and clock data. The class inherits from the `IParse` base class and implements the `_parse` method tailored for SP3 GPS files.

Usage:
    To use this module, instantiate an object of the `IParseSP3GPS` class and call the `_parse` method with the filename of the SP3 file to extract GPS position and clock data.

Example:
    ```
    from your_module import IParseSP3GPS

    parser = IParseSP3GPS()
    position_data, clock_data = parser._parse('sample.sp3')
    ```
"""

from pathlib import Path
from typing import Tuple

import georinex as gr
import pandas as pd
from pandas.core.api import DataFrame as DataFrame
from pandas.core.api import Series as Series

from ..base_iparse import IParse


class IParseSP3GPS(IParse):
    """This is an Interface class for parsing SP3 files.

    This class inherits from IParse and implements the _parse method
    specifically for SP3 GPS files.

    Attributes:
        Inherits all attributes from the IParse base class.

    Methods:
        parse: Parses the SP3 file to extract GPS position and clock data.

    Example:
        Instantiate an object of IParseSP3GPS and call the _parse method with
        the filename of the SP3 file to extract GPS position and clock data.

        ```
        parser = IParseSP3GPS()
        position_data, clock_data = parser.parse('sample.sp3')
        ```
    """

    def __init__(self) -> None:
        """Initializes the IParseSP3GPS class."""
        super().__init__("SP3-GPS")

    def parse(self, filename: Path, **kwargs) -> Tuple[Series, DataFrame]:
        """Parses the SP3 file to extract GPS position and clock data.

        Args:
            filename: The filename of the SP3 file.
            **kwargs: Additional keyword arguments to be passed to the georinex.load_sp3 function.

        Returns:
            Tuple[Series, DataFrame]: A tuple containing the metadata and data extracted from the SP3 file.
        """
        # Read SP3 file
        df = (
            gr.load_sp3(fn=filename, outfn=None, **kwargs).to_dataframe().reset_index(2)
        )

        # Pivot the dataframe with ECEF coordinates
        df = df.pivot(columns="ECEF")
        # Get the position table
        pos = df["position"]
        clock = df["clock"][["x"]].rename({"x": "clock"}, axis=1)
        dclock = df["dclock"][["x"]].rename({"x": "dclock"}, axis=1)

        # Join the position and clock tables
        return pd.Series(), pd.concat([pos, clock, dclock], axis=1)
