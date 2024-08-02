"""Interface for GPS observational data parsing from RINEX files.

This class defines an interface for parsing GPS observational data from RINEX (Receiver Independent Exchange) files.


Attributes:
    features (str): A string indicating the type of data to parse ('gps_nav' for GPS navigation data).

Methods:
    parse(filename: str | Path) -> Tuple[pd.Series, pd.DataFrame]:
        Parses GPS observational data from a RINEX file and returns it as a Pandas DataFrame along with associated metadata.

Args:
        filename (str | Path): The path to the RINEX file to parse.

Returns:
        Tuple[pd.Series, pd.DataFrame]: A tuple containing the metadata as a Pandas Series and the parsed data as a Pandas DataFrame.

Example Usage:
    ```
    parser = IParseGPSObs()
    metadata, parsed_data = parser.parse('/path/to/rinex_file.19o')
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

    L1_CODE_ON = "C1C"
    L1_PHASE_ON = "L1C"
    L2_CODE_ON = "C2W"
    L2_PHASE_ON = "L2W"

    def __init__(
        self,
    ) -> None:
        """Initialize the class."""
        super().__init__(features="gps_obs")

    def _dispatch_rinex2(
        self, filename: str | Path, **kwargs
    ) -> tp.Tuple[pd.Series, pd.DataFrame]:
        """Dispatches the parsing of a RINEX 2 file to the appropriate method.

        This method dispatches the parsing of a RINEX 2 file to the appropriate method based on the file's contents.

        Args:
            filename (str | Path): The path to the RINEX file to parse.
            kwargs: Additional keyword arguments to pass to the parser.

        Returns:
            Tuple[pd.Series, pd.DataFrame]: A tuple containing the metadata as a Pandas Series and the parsed data as a Pandas DataFrame.
        """
        # Open the RINEX navigation file using the `georinex` library
        # Returns as a `xarray.Dataset`
        rinex_data = gr.load(
            rinexfn=filename,
            use="G",
            fast=True,
            **kwargs,
        )

        # Convert the `xarray.Dataset` to a `pandas.DataFrame` and metadata to a `pandas.Series`
        rinex_data = rinex_data.to_dataframe()

        # Add the header information to the metadata
        metadata = pd.Series(gr.rinexheader(fn=filename))

        # Filter the dataframme to drop all null rows
        rinex_data.dropna(axis=1, thresh=4, inplace=True)

        # Return the metadata and data
        return metadata, rinex_data

    def _dispatch_rinex3(
        self, filename: str | Path, **kwargs
    ) -> tp.Tuple[pd.Series, pd.DataFrame]:
        """Dispatches the parsing of a RINEX 3 file to the appropriate method.

        This method dispatches the parsing of a RINEX 3 file to the appropriate method based on the file's contents.

        Args:
            filename (str | Path): The path to the RINEX file to parse.
            kwargs: Additional keyword arguments to pass to the parser.

        Returns:
            Tuple[pd.Series, pd.DataFrame]: A tuple containing the metadata as a Pandas Series and the parsed data as a Pandas DataFrame.
        """
        # Open the RINEX navigation file using the `georinex` library
        # Returns as a `xarray.Dataset`
        rinex_data = gr.load(
            rinexfn=filename,
            use="G",
            **kwargs,
        )

        # Convert the `xarray.Dataset` to a `pandas.DataFrame` and metadata to a `pandas.Series`
        rinex_data = rinex_data.to_dataframe()

        # Add the header information to the metadata
        metadata = pd.Series(gr.rinexheader(fn=filename))

        # Return the metadata and data
        return metadata, rinex_data

    def parse(
        self, filename: str | Path, **kwargs
    ) -> tp.Tuple[pd.Series, pd.DataFrame]:
        """Parse GPS observational data from a RINEX file.

        This method parses GPS observational data from a RINEX file using the 'georinex' library, converts it to a
        Pandas DataFrame, and returns both the metadata and the parsed data.

        Args:
            filename (str | Path): The path to the RINEX file to parse.
            kwargs: Additional keyword arguments to pass to the parser.

        Returns:
            Tuple[pd.Series, pd.DataFrame]: A tuple containing the metadata as a Pandas Series and the parsed data as a Pandas DataFrame.
        """
        # Open the header of the RINEX file to determine the version
        rinex_version = self.get_rinex_version(filename)

        # Dispatch the parsing to the appropriate method based on the RINEX version
        if str(rinex_version).startswith("2"):
            return self._dispatch_rinex2(filename, **kwargs)

        return self._dispatch_rinex3(filename, **kwargs)

    @staticmethod
    def _multiindex_interpolator(
        df: pd.DataFrame, method: str, order: int | None = None, **kwargs
    ) -> pd.DataFrame:
        """Helper method to interpolate missing values in a DataFrame with a MultiIndex.

        Args:
            df (pd.DataFrame): The DataFrame to interpolate.
            method (str): The method to use for interpolation.
            order (int | None): The order of the interpolation method.
            **kwargs: Additional keyword arguments to pass to the interpolation method.

        Returns:
            pd.DataFrame: The DataFrame with missing values interpolated.
        """
        index = df.index
        # Drop the index
        df = df.reset_index(drop=True)
        # Interpolate the missing values
        # NOTE: This is due to inconsistency in the `pandas` library when interpolating with a MultiIndex
        df = df.interpolate(method=method, order=order, **kwargs)

        return df.set_index(index)

    @staticmethod
    def interpolate_missing_values(
        obsData: pd.DataFrame,
        method: str = "from_derivatives",
        order: int | None = None,
        subset: tp.List[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Interpolate missing values range measurements in a DataFrame.

        This method interpolates missing values in a DataFrame using the specified method.

        Args:
            obsData (pd.DataFrame): The DataFrame to interpolate.
            method (str): The method to use for interpolation. Default is 'from_derivatives'.
            order (int | None): The order of the interpolation method.
            **kwargs: Additional keyword arguments to pass to the interpolation method.

        Returns:
            pd.DataFrame: The DataFrame with missing values interpolated.
        """
        return (
            obsData.reorder_levels(["sv", "time"])
            .groupby(level="sv")
            .apply(
                lambda x: IParseGPSObs._multiindex_interpolator(
                    x, method=method, order=order, **kwargs
                )
            )
            .droplevel(0)
            .reorder_levels(["time", "sv"])
            .sort_index()
            .dropna(subset=subset, how="any")
        )

    @staticmethod
    def get_rinex_version(filename: str | Path) -> str:
        """Get the RINEX version of a file.

        This method reads the header of a RINEX file and returns the version number.

        Args:
            filename (str | Path): The path to the RINEX file to parse.

        Returns:
            str: The version number of the RINEX file.
        """
        return gr.rinexinfo(filename)["version"]

    @staticmethod
    def get_timestamps(filename: str | Path) -> pd.Series:
        """Get the timestamps of a RINEX file.

        This method reads the timestamps from a RINEX file and returns them as a Pandas Series.

        Args:
            filename (str | Path): The path to the RINEX file to parse.

        Returns:
            pd.Series: The timestamps of the RINEX file.
        """
        return pd.to_datetime(gr.gettime(filename))
