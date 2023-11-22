"""Mounts the FTP server to the local file system."""
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

__all__ = ["AbstractMountServer", "CDDISMountServer"]


class AbstractMountServer(ABC):
    """An abstract class for mounting an FTP server to the local file system.

    This class provides a blueprint for mounting and unmounting FTP servers to the local directory.

    Attributes:
        _mountDir (Path): The path to the mount directory.
        _mounted (bool): A flag indicating if the FTP server is mounted.
    """

    def __init__(self, mountDir: Path) -> None:
        """Initializes the AbstractMountServer instance.

        Args:
            mountDir (Path): Path to the mount directory.
        """
        # Check if the mount directory is valid
        self._validate_mount_dir(mountDir)

        self._mountDir = mountDir
        self._is_mounted = False

    def _validate_mount_dir(self, mountPoint: Path) -> None:
        """Validates the mount directory.

        Has to be empty if it exists.
        """
        if mountPoint.exists() and not mountPoint.is_dir():
            raise ValueError("Mount point is not a directory")

        if mountPoint.exists() and len(os.listdir(mountPoint)) > 0:
            raise ValueError("Mount point is not empty")

        return

    def _make_mount_point(self) -> None:
        """Creates the mount point directory."""
        if not self._mountDir.exists():
            os.makedirs(self._mountDir)

    def _tear_down_mount_point(self) -> None:
        """Tears down the mount point directory."""
        if self._mountDir.exists() and not self.isMounted:
            os.rmdir(self._mountDir)

    def mount(self) -> None:
        """Mounts the FTP server to the local file system."""
        # If already mounted, return
        if self.isMounted:
            return

        # Make the mount point directory
        self._make_mount_point()

        # Mount the ftp server
        self._mountLogic()

        # Set the mounted flag
        self._is_mounted = True

        return

    def unmount(self) -> None:
        """Unmounts the FTP server from the local file system."""
        # If not mounted, return
        if not self.isMounted:
            return

        # Unmount the ftp server
        self._unmountLogic()
        # Set the mounted flag
        self._is_mounted = False

        # Tear down the mount point directory
        self._tear_down_mount_point()

        return

    @abstractmethod
    def _mountLogic(self) -> None:
        """Mount logic."""
        pass

    @abstractmethod
    def _unmountLogic(self) -> None:
        """Unmount logic."""
        pass

    @property
    def mountDir(self) -> Path:
        """Returns the mount directory."""
        return self._mountDir

    @mountDir.setter
    def mountDir(self, mountDir: Path) -> None:
        """Sets the mount directory."""
        raise AttributeError(
            "Cannot set mount directory. Use the constructor to set the mount directory."
        )

    @property
    def isMounted(self) -> bool:
        """Returns True if the FTP server is mounted."""
        return self._is_mounted

    @isMounted.setter
    def isMounted(self, mounted: bool) -> None:
        """Sets the mounted flag."""
        raise AttributeError(
            "Cannot set mounted flag. Use the mount and unmount methods to set the mounted flag."
        )


class CDDISMountServer(AbstractMountServer):
    """A class for mounting the CDDIS FTP server to the local file system.

    This class extends AbstractMountServer to specifically handle CDDIS FTP server mounting
    and unmounting operations.

    Attributes:
        _email (str): Email address used for anonymous login.
    """

    def __init__(self, mountDir: Path, email: str) -> None:
        """Initializes the CDDISMountServer instance.

        Args:
            mountDir (Path): Path to the mount directory.
            email (str): Email address to use for anonymous login.


        Raises:
            Exception: If curlftpfs is not installed.
            ValueError: If the provided email is not valid.
        """
        if not self._check_curftpfs():
            raise Exception("curlftpfs is not installed")

        if "@" not in email:
            raise ValueError("Email is not valid")

        self._email = email
        super().__init__(mountDir)

    def _mountLogic(self) -> None:
        """Mounts the CDDIS FTP server."""
        mntCmd = f"curlftpfs -o ssl -o user='anonymous:{self._email}' gdc.cddis.eosdis.nasa.gov {self.mountDir}"

        if os.system(mntCmd):
            raise Exception("Error mounting the ftp server")

        return

    def _unmountLogic(self) -> None:
        """Unmounts the CDDIS FTP server."""
        umntCmd = f"umount -l {self.mountDir}"

        if os.system(umntCmd):
            print("Error unmounting the ftp server")
            print(
                f"Manually unmount the ftp server using the following command: {umntCmd}"
            )
            raise Exception("Error unmounting the ftp server")

    def _check_curftpfs(self) -> bool:
        """Check if curlftpfs is installed in the system."""
        return shutil.which("curlftpfs") is not None
