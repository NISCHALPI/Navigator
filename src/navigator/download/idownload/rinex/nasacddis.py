"""Nasa CDDIS Rinex Downloader."""


import random
from datetime import datetime
from ftplib import FTP_TLS
from pathlib import Path

from ..idownload import IDownload

__all__ = ["NasaCDDIS"]

# Source: https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/hourly_30second_data.html


class NasaCDDIS(IDownload):
    """Nasa CDDIS Rinex Downloader."""

    host = "gdc.cddis.eosdis.nasa.gov"
    username = "anonymous"
    email = "jhondoe@gmail.com"
    directory = "/gnss/data/hourly/"
    format = "crx"

    def __init__(self) -> None:
        super().__init__("NasaCDDIS")
        self._establish_ftp_connection()

    def _establish_ftp_connection(self) -> FTP_TLS:
        """_summary_.

        Returns:
            FTP_TLS: _description_
        """
        ftps = FTP_TLS(host=self.host)
        ftps.login(user=self.username, passwd=self.email)
        ftps.prot_p()
        ftps.cwd(self.directory)
        self.ftps = ftps
        return ftps

    def _check_kwargs(self, *args, **kwargs) -> None:  # noqa
        """Check kwargs for essential arguments."""
        if "station" not in kwargs:
            raise ValueError(
                """station_name must be provided as kwarg.
                             See : https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/hourly_30second_data.html"""
            )
        if "data_type" not in kwargs:
            raise ValueError("data_type must be provided as kwarg")

        if "start_time" not in kwargs:
            raise ValueError("start_time must be provided as kwarg")

        if not isinstance(kwargs["start_time"], datetime):
            raise ValueError("start_time must be datetime object")

        if "save_path" not in kwargs:
            raise ValueError("save_path must be provided as kwarg")

        if "country_code" not in kwargs:
            raise ValueError(
                """country_code must be provided as kwarg.
                             See : https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/hourly_30second_data.html"""
            )
        if "marker" not in kwargs:
            raise ValueError(
                """marker must be provided as kwarg.
                             See : https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/hourly_30second_data.html"""
            )

        if "reciever_number" not in kwargs:
            raise ValueError(
                """reciever_number must be provided as kwarg.
                             See : https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/hourly_30second_data.html"""
            )

        return

    def _format_codes(self, *args, **kwargs) -> str:  # noqa
        """_summary_.

        Returns:
            None: _description_
        """
        start_time: datetime = kwargs["start_time"]

        # Four digit year
        year = str(start_time.year)
        # Three digit day of year
        day = str(start_time.timetuple().tm_yday).zfill(3)
        #  2 Digit hour of day
        hour = str(start_time.hour).zfill(2)

        # 2digit minute of hour
        minute = str(start_time.minute).zfill(2)

        # Station name
        station = kwargs["station"]

        # Get random number between 0 and 9
        marker = kwargs["marker"]

        # ISO  contry code
        country_code = kwargs["country_code"]

        # Data source
        K = "R"  # see https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/hourly_30second_data.html

        # Data type
        data_type = kwargs["data_type"]

        # File format
        file_format = self.format

        # Compresion
        compression = "gz"

        filename = f"{station}{marker}{marker}{country_code}_{K}_{year}{day}{hour}{minute}_01H_30S_{data_type}.{file_format}.{compression}"

        # Make a format code string
        return self.directory + f"""{year}/{day}/{hour}/{filename}""", filename

    def _download(self, *args, **kwargs) -> None:
        """_summary_.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        # Check kwargs for essential arguments
        self._check_kwargs(*args, **kwargs)

        # Get format codes and filename
        ftp_path, filename = self._format_codes(*args, **kwargs)

        print(f"Downloading {filename} from {ftp_path}")

        # Save path
        savepath = Path(kwargs["save_path"]) / filename

        # Download file
        with open(savepath, "wb") as f:
            self.ftps.retrbinary(f"RETR {ftp_path}", f.write)

        return
