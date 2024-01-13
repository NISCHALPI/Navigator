"""Interface for GPS navigation data parsing from RINEX files.

This class defines an interface for parsing GPS navigation data from RINEX (Receiver Independent Exchange) files.

Attributes:
    features (str): A string indicating the type of data to parse ('gps_nav' for GPS navigation data).

Methods:
    parse(filename: str | Path) -> Tuple[pd.Series, pd.DataFrame]: 
        Parses GPS navigation data from a RINEX file and returns it as a Pandas DataFrame along with associated metadata.

Args:
        filename (str | Path): The path to the RINEX file to parse.

Returns:
        Tuple[pd.Series, pd.DataFrame]: A tuple containing the metadata as a Pandas Series and the parsed data as a Pandas DataFrame.

Example Usage:
    ```
    parser = IParseGPSNav()
    metadata, parsed_data = parser.parse('/path/to/rinex_nav_file.19n')
    ```

Note:
    This class should be subclassed to implement specific parsing behavior for different data types.
"""


import typing as tp
from pathlib import Path

import georinex as gr
import pandas as pd

from ..base_iparse import IParse

__all__ = ["IParseGPSNav"]


class IParseGPSNav(IParse):
    """Interface for GPS navigation data parsing from RINEX files."""

    def __init__(self) -> None:
        """Initialize a new instance of the IParseGPSNav class.

        This constructor initializes the IParseGPSNav class and sets the 'features' attribute to 'gps_nav' to indicate
        that it is designed for parsing GPS navigation data.
        """
        super().__init__(features="gps_nav")

    def parse(self, filename: str | Path) -> tp.Tuple[pd.Series, pd.DataFrame]:
        """Parse GPS navigation data from a RINEX file.

        This method parses GPS navigation data from a RINEX file using the georinex parsing backend, converts it to a
        Pandas DataFrame, and returns both the metadata and the parsed data.

        Args:
            filename (str | Path): The path to the RINEX file to parse.

        Returns:
            Tuple[pd.Series, pd.DataFrame]: A tuple containing the metadata as a Pandas Series and the parsed data as a Pandas DataFrame.
        """
        # Open the RINEX navigation file using the `georinex` library
        # Returns as a `xarray.Dataset`
        rinex_data = gr.rinexnav(filename, use="G")

        # Convert the `xarray.Dataset` to a `pandas.DataFrame` and metadata to a `pandas.Series`
        rinex_data = rinex_data.to_dataframe()
        metadata = pd.Series(gr.rinexheader(fn=filename))

        # Filter the dataframme to drop all null rows
        rinex_data = rinex_data.dropna(axis=0, thresh=10)

        # Return the metadata and data
        return metadata, rinex_data
