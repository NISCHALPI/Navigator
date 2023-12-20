"""Implementation of the FTPFS server to be used by the download module."""
from pathlib import Path

from fs.ftpfs import FTPFS

__all__ = ["FTPFSServer"]


class FTPFSServer:
    """Implementation of the FTPFS server to be used by the download module.

    This class establishes a connection to an FTP server and provides methods
    to interact with the server, check its availability, and download files.

    Attributes:
        host (str): The FTP server hostname or IP address.
        user (str): The username for authentication.
        acct (str): The account name for authentication.
        tls (bool, optional): Indicates whether TLS encryption should be used. Defaults to True.

    Methods:
        __init__(self, host: str, user: str, acct: str, tls: bool = True) -> None:
            Initializes the FTPFSServer with the provided host, user, acct, and tls settings.

        connect(self) -> None:
            Establishes a connection to the FTP server using the specified credentials.

        is_alive(self) -> bool:
            Checks if the FTP server connection is alive by attempting to list the root directory.

        is_available(self, ftp_file_path: str) -> bool:
            Checks if a file or directory specified by ftp_file_path exists on the server.

        download(self, ftp_file_path: str, save_path: Path) -> None:
            Downloads a file from the FTP server identified by ftp_file_path and saves it to the local filesystem
            at the specified save_path.
    """

    def __init__(self, host: str, user: str, acct: str, tls: bool = True) -> None:
        """Initialize the FTPFSServer.

        Args:
            host (str): The FTP server hostname or IP address.
            user (str): The username for authentication.
            acct (str): The account name for authentication.
            tls (bool, optional): Indicates whether TLS encryption should be used. Defaults to True.
        """
        self.host = host
        self.user = user
        self.acct = acct
        self.tls = tls

    def connect(self) -> None:
        """Establish a connection to the FTP server using the specified credentials."""
        self.fs = FTPFS(
            host=self.host,
            user=self.user,
            acct=self.acct,
            tls=self.tls,
        )
        return

    def is_alive(self) -> bool:
        """Check if the FTP server connection is alive by attempting to list the root directory.

        Returns:
            bool: True if the connection is alive, False otherwise.
        """
        try:
            # Attempt to list the root directory
            self.fs.listdir("/")
            return True
        except Exception:
            return False

    def is_available(self, ftp_file_path: str) -> bool:
        """Check if a file or directory specified by ftp_file_path exists on the server.

        Args:
            ftp_file_path (str): The path of the file or directory on the server.

        Returns:
            bool: True if the file or directory exists, False otherwise.
        """
        if not self.is_alive():
            self.connect()

        return self.fs.exists(ftp_file_path)

    def download(self, ftp_file_path: str, save_path: str) -> None:
        """Download a file from the FTP server and save it to the local filesystem.

        Args:
            ftp_file_path (str): The path of the file on the FTP server.
            save_path (str): The path where the file should be saved on the local filesystem.
        """
        # If not connected, connect to the server
        if not self.is_alive():
            self.connect()

        # Get the file extension from the file path
        fname = Path(ftp_file_path).name

        # Open the file in binary mode
        with open(Path(save_path) / fname, "wb") as f:
            # Copy the file from the FTP server to the local file
            self.fs.download(ftp_file_path, f)

    def listdir(self, ftp_dir_path: str) -> list:
        """List the contents of a directory on the FTP server.

        Args:
            ftp_dir_path (str): The path of the directory on the FTP server.

        Returns:
            list: A list of the contents of the directory.
        """
        # If not connected, connect to the server
        if not self.is_alive():
            self.connect()

        # List the contents of the directory
        return self.fs.listdir(ftp_dir_path)

    def close(self) -> None:
        """Close the connection to the FTP server."""
        if self.is_alive():
            self.fs.close()
        return
