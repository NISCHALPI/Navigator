"""Ublox Receiver Streaming Profile Module.

This module defines the StreamingProfile class, which provides methods for controlling Ublox receivers
through serial communication streams. The StreamingProfile class is designed to collect and manage data
from the receiver, including navigation solution data and RAWX observation data for RINEX conversion.
It also includes functionality for sending commands to configure the receiver's data collection behavior.

Classes:
    StreamingProfile: Manages the data collection and command communication with the Ublox receiver.

Constants:
    COLLECTION_TOUT: A timeout value used for data collection operations.
"""

import typing as tp
from logging import Logger
from pathlib import Path

import pyubx2 as ubx
from serial import Serial

from .commands import BaseCommand
from .controller import Controller

__all__ = ["StreamingProfile"]

COLLECTION_TOUT = 1.2


class StreamingProfile:
    """StreamingProfile class for the Ublox receiver.

    This class provides methods for controlling Ublox receivers through Serial communication streams.
    The streaming profile is used to collect two types of data:
    - Navigation solution data
    - RAWX observation data for RINEX conversion

    Attributes:
        PVT_MESSAGE_CMD: Command to set the PVT message rate.
        RAWX_MESSAGE_CMD: Command to set the RAWX message rate.
        serial_port (Serial): The serial port for communication with the receiver.
        controller (Controller): The controller object to manage communication and data collection.

    Methods:
        __init__(device, baudrate=115200, timeout=3, logger=None, no_check=False): Initializes the StreamingProfile.
        send_collection_command(): Sends the collection command to the controller.
        collect(n): Collects data from the receiver.
        start(): Starts the collection of data.
        stop(): Stops the collection of data.
    """

    def __init__(
        self,
        commands: tp.List[BaseCommand],
        device: Path,
        baudrate: int = 115200,
        timeout: int = 3,
        logger: tp.Optional[Logger] = None,
        no_check: bool = False,
    ) -> None:
        """Initializes the StreamingProfile object.

        Args:
            commands (List[BaseCommand]): The list of commands to send to the receiver.
            device (Path): The path to the device file.
            baudrate (int, optional): The baudrate of the serial connection. Defaults to 115200.
            timeout (int, optional): The timeout of the serial connection. Defaults to 3.
            logger (Optional[Logger], optional): The logger object to log I/O data. Defaults to None.
            no_check (bool, optional): If True, disables all default message rates. Defaults to False.

        Raises:
            ValueError: If the device file does not exist.

        Note:
            The device file must exist and should be something like "/dev/ttyACM0".
        """
        # Store the commands list
        self.commands = commands
        # Create a serial port
        self.serial_port = Serial(str(device), baudrate, timeout=timeout)
        # Create a controller object
        self.controller = Controller(self.serial_port, logger=logger, no_check=no_check)

    def send_message_recording_command(self) -> None:
        """Sends the collection command to the controller.

        Returns:
            None
        """
        # Send the collection command to activate the data collection
        for command in self.commands:
            self.controller.send_config_command(command=command.config_command())

    def collect(
        self,
        parse: bool = False,
    ) -> tuple[
        int,
        dict[str, list[ubx.UBXMessage]],
    ]:
        """Collects data from the receiver.

        Args:
            n (int): The number of messages to collect. Defaults to -1.
            parse (bool): If True, the messages are parsed. Defaults to False.

        Returns:
            tuple[int, dict[str, list[ubx.UBXMessage]]]: The number of messages collected and the messages.
        """
        get_io_msg = self.controller.flush_messages()
        # Make a dictionary to store the messages
        cmd_data_map = {str(cmd): [] for cmd in self.commands}

        # Store the messages in the dictionary
        for msg in get_io_msg:
            # Change "-" to "_" for comaptiblaity NAV-POSLLH" -> "NAV_POSLLH"
            for cmd in self.commands:
                if cmd == msg.identity:
                    cmd_data_map[msg.identity].append(
                        cmd.parse_ubx_message(msg) if parse else msg
                    )
        return (int(len(get_io_msg)), cmd_data_map)

    def start(self) -> None:
        """Starts the collection of data.

        Returns:
            None
        """
        # Send the collection command
        self.send_message_recording_command()

    def stop(self) -> None:
        """Stops the collection of data.

        Returns:
            None
        """
        # Stop the collection of data
        self.controller.stop()

        # Close the serial port
        self.serial_port.close()
