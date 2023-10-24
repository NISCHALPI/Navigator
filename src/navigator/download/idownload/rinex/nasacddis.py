"""Navigator Nasa CDDIS Rinex Downloader."""
from datetime import datetime
from ftplib import FTP_TLS
from pathlib import Path

from ..idownload import IDownload


class NasaCDDIS(IDownload):
    """Nasa CDDIS Rinex Downloader.

    This class provides a way to download RINEX data from the NASA CDDIS FTP server.

    Attributes:
        HOST (str): The FTP server hostname.
        USERNAME (str): The FTP server username (anonymous for NASA CDDIS).
        EMAIL (str): The email address used as the password for anonymous login.
        DIRECTORY (str): The base directory on the FTP server where RINEX data is stored.
        FORMAT (str): The default RINEX data format.

    Methods:
        __init__(): Initialize the NasaCDDIS class.
        _establish_ftp_connection(): Establish a secure FTP connection to the NASA CDDIS server.
        _check_kwargs(**kwargs): Check if essential keyword arguments are provided.
        _format_codes(**kwargs): Format the codes for the RINEX file.
        _download(**kwargs): Download the RINEX file from the NASA CDDIS server.

     RINEX V3 file name codes format:

    - YYYY: 4-digit year
    - DDD: 3-digit day of year
    - HH: 2-digit hour of day (00, 01, ..., 23)
    - XXXX: 4-character IGS station name
    - M: Monument or marker number (0-9)
    - R: Receiver number (0-9)
    - CCC: ISO country code
    - tt: Type of data

    Type of data codes:
    - GO: GPS Observation data
    - RO: GLONASS Observation data
    - EO: Galileo Observation data
    - JO: QZSS Observation data
    - CO: BDS Observation data
    - IO: IRNSS Observation data
    - SO: SBAS Observation data
    - MO: Mixed Observation data (doi:10.5067/GNSS/gnss_hourly_o_001)
    - GN: GPS Navigation data (doi:10.5067/GNSS/gnss_hourly_n_001)
    - RN: GLONASS Navigation data (doi:10.5067/GNSS/gnss_hourly_g_001)
    - EN: Galileo Navigation data (doi:10.5067/GNSS/gnss_hourly_l_001)
    - JN: QZSS Navigation data (doi:10.5067/GNSS/gnss_hourly_q_001)
    - CN: BDS Navigation data (doi:10.5067/GNSS/gnss_hourly_f_001)
    - IN: IRNSS Navigation data (doi:10.5067/GNSS/gnss_hourly_i_001)
    - SN: SBAS Navigation data (doi:10.5067/GNSS/gnss_hourly_h_001)
    - MN: Navigation data (All GNSS Constellations) (doi:10.5067/GNSS/gnss_hourly_x_001)
    - MM: Meteorological Observation (doi:10.5067/GNSS/gnss_hourly_m_001)

    - FFF: File format
        - rnx: RINEX
        - crx: Hatanaka Compressed RINEX
        - .gz: Compressed file
    """

    HOST = "gdc.cddis.eosdis.nasa.gov"
    USERNAME = "anonymous"
    EMAIL = "jhondoe@gmail.com"
    DIRECTORY = "/gnss/data/hourly/"
    FORMAT = "crx"

    def __init__(self) -> None:
        """Initialize the NasaCDDIS class."""
        super().__init__("NasaCDDIS")
        self._establish_ftp_connection()

    def _establish_ftp_connection(self) -> None:
        """Establish a secure FTP connection to the NASA CDDIS server."""
        ftps = FTP_TLS(self.HOST)
        ftps.login(user=self.USERNAME, passwd=self.EMAIL)
        ftps.prot_p()
        ftps.cwd(self.DIRECTORY)
        self.ftps = ftps

    def _check_kwargs(self, **kwargs) -> None:
        """Check kwargs for essential arguments.

        Args:
            kwargs (dict): Keyword arguments to be checked.

        Raises:
            ValueError: If any of the essential kwargs are missing or invalid.
        """
        essential_kwargs = [
            "station",
            "data_type",
            "start_time",
            "save_path",
            "country_code",
            "marker",
            "receiver_number",
        ]
        for arg in essential_kwargs:
            if arg not in kwargs:
                raise ValueError(f"{arg} must be provided as a kwarg.")
        if not isinstance(kwargs["start_time"], datetime):
            raise ValueError("start_time must be a datetime object.")

    def _format_codes(self, **kwargs) -> tuple:
        """Format the codes for the RINEX file.

        Args:
            kwargs (dict): Keyword arguments for formatting the RINEX file codes.

        Returns:
            tuple: A tuple containing the FTP path and the filename.
        """
        start_time = kwargs["start_time"]
        year = str(start_time.year)
        day = str(start_time.timetuple().tm_yday).zfill(3)
        hour = str(start_time.hour).zfill(2)
        minute = str(start_time.minute).zfill(2)
        station = kwargs["station"]
        marker = kwargs["marker"]
        country_code = kwargs["country_code"]
        data_type = kwargs["data_type"]
        file_format = self.FORMAT
        compression = "gz"
        K = "R"
        filename = f"{station}{marker}{marker}{country_code}_{K}_{year}{day}{hour}{minute}_01H_30S_{data_type}.{file_format}.{compression}"
        return self.DIRECTORY + f"{year}/{day}/{hour}/{filename}", filename

    def _download(self, **kwargs) -> None:
        """Download the RINEX file from the NASA CDDIS server.

        Args:
            kwargs (dict): Keyword arguments for specifying download parameters.
        """
        # Check kwargs for essential arguments
        self._check_kwargs(**kwargs)

        # Format the codes for the RINEX file
        ftp_path, filename = self._format_codes(**kwargs)

        print(f"Downloading {filename} from {ftp_path}")
        # Download the RINEX file
        savepath = Path(kwargs["save_path"]) / filename
        with open(savepath, "wb") as f:
            self.ftps.retrbinary(f"RETR {ftp_path}", f.write)
