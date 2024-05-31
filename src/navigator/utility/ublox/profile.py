"""The profiles to run the Ublox receiver in i.e collect data and send commands to the receiver in a defined manner."""

import typing as tp
from logging import Logger
from pathlib import Path

import pyubx2 as ubx
from serial import Serial

from .controller import Controller

__all__ = ["StreamingProfile"]

COLLECTION_TOUT = 1.2


class StreamingProfile:
    """StreamingProfile class for the Ublox receiver.

    This class provides methods for controlling Ublox receivers through Serial communication streams.
    The streaming profile is used to collect two types of data:
    - Navigation solution data
    - RAWX observation data for rinex conversion.


    """

    # COMMAND TO SET THE PVT MESSAGE RATE
    PVT_MESSAGE_CMD = ubx.UBXMessage.config_set(
        layers=ubx.SET_LAYER_RAM,
        transaction=ubx.TXN_NONE,
        cfgData=[("CFG_MSGOUT_UBX_NAV_PVT_USB", 1)],
    )
    RAWX_MESSAGE_CMD = ubx.UBXMessage.config_set(
        layers=ubx.SET_LAYER_RAM,
        transaction=ubx.TXN_NONE,
        cfgData=[("CFG_MSGOUT_UBX_RXM_RAWX_USB", 1)],
    )

    def __init__(
        self,
        device: Path,
        baudrate: int = 115200,
        timeout: int = 3,
        logger: tp.Optional[Logger] = None,
        no_check: bool = False,
    ) -> None:
        """Initializes the StreamingProfile object.

        Args:
            device (Path): The path to the device file.
            baudrate (int, optional): The baudrate of the serial connection. Defaults to 115200.
            timeout (int, optional): The timeout of the serial connection. Defaults to 3.
            logger (Optional[Logger], optional): The logger object to log I/O data. Defaults to None.
            no_check (bool, optional): If True, disables all default message rates. Defaults to False.

        Raises:
            ValueError: If the device file does not exist.

        Returns:
            None

        Note:
            - The device file must exist and should be something like "/dev/ttyACM0".
        """
        # Create a serial port
        self.serial_port = Serial(str(device), baudrate, timeout=timeout)
        # Create a controller object
        self.controller = Controller(self.serial_port, logger=logger, no_check=no_check)

    def send_collection_command(self) -> None:
        """Sends the collection command to the controller.

        Returns:
            None
        """
        # Send the collection command
        self.controller.send_config_command(self.PVT_MESSAGE_CMD)
        self.controller.send_config_command(self.RAWX_MESSAGE_CMD)

        return

    def collect(
        self, n: int
    ) -> tp.Tuple[tp.List[ubx.UBXMessage], tp.List[ubx.UBXMessage]]:
        """Collects data from the receiver.

        Args:
            n (int): The number of messages to collect.

        Returns:
            Tuple[List[UBXMessage], List[UBXMessage]]: The collected navigation and rinex data.
        """
        # Acquire the io lock from the controller
        nav_msg = []
        rinex_msg = []
        get_io_msg = self.controller.flush_n_messages(n)

        for msg in get_io_msg:
            if msg.identity == "NAV-PVT":
                nav_msg.append(msg)
            elif msg.identity == "RXM-RAWX":
                rinex_msg.append(msg)

            continue

        return nav_msg, rinex_msg

    def start(self) -> None:
        """Starts the collection of data.

        Returns:
            None
        """
        # Send the collection command
        self.send_collection_command()
        return

    def stop(self) -> None:
        """Stops the collection of data.

        Returns:
            None
        """
        # Stop the collection of data
        self.controller.stop()
        # Close the serial port
        self.serial_port.close()
        return
