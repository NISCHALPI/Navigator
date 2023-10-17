"""Interface for GPS observational data parsing from RINEX files.

This class defines an interface for parsing GPS observational data from RINEX (Receiver Independent Exchange) files.


Attributes:
    features (str): A string indicating the type of data to parse ('gps_nav' for GPS navigation data).

Methods:
    _parse(filename: str | Path) -> Tuple[pd.Series, pd.DataFrame]: 
        Parses GPS observational data from a RINEX file and returns it as a Pandas DataFrame along with associated metadata.

Args:
        filename (str | Path): The path to the RINEX file to parse.

Returns:
        Tuple[pd.Series, pd.DataFrame]: A tuple containing the metadata as a Pandas Series and the parsed data as a Pandas DataFrame.

Example Usage:
    ```
    parser = IParseGPSObs()
    metadata, parsed_data = parser._parse('/path/to/rinex_file.19o')
    ```

Note:
    This class should be subclassed to implement specific parsing behavior for different data types.
"""


import typing as tp
from pathlib import Path

import georinex as gr
import pandas as pd

from ..base_iparse import IParse

__all__ = ["IParseGPSObs"]


class IParseGPSObs(IParse):
    """Interface for GPS observational data parsing from RINEX files."""

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__(features="gps_obs")

    def _parse(self, filename: str | Path) -> tp.Tuple[pd.Series, pd.DataFrame]:
        """Parse GPS observational data from a RINEX file.

        This method parses GPS observational data from a RINEX file using the 'georinex' library, converts it to a
        Pandas DataFrame, and returns both the metadata and the parsed data.

        Args:
            filename (str | Path): The path to the RINEX file to parse.

        Returns:
            Tuple[pd.Series, pd.DataFrame]: A tuple containing the metadata as a Pandas Series and the parsed data as a Pandas DataFrame.
        """
        # Open the RINEX navigation file using the `georinex` library
        # Returns as a `xarray.Dataset`
        rinex_data = gr.rinexobs3(fn=filename, use="G")

        # Convert the `xarray.Dataset` to a `pandas.DataFrame` and metadata to a `pandas.Series`
        rinex_data = rinex_data.to_dataframe()

        # Add the header information to the metadata
        metadata = pd.Series(gr.rinexheader(fn=filename))

        # Filter the dataframme to drop all null rows
        rinex_data = rinex_data.dropna(axis=0, thresh=10)

        # Return the metadata and data
        return metadata, rinex_data
