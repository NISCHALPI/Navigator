"""Controller module for the Ublox receiver.

This module provides a Controller class for interfacing with Ublox receivers through either Serial or socket communication streams.

Classes:
    Controller: Main class for controlling Ublox receivers.

Attributes:
    __all__ (List[str]): List of symbols exported by this module.
    DEFAULT_COMMAND_WAIT_TIME (float): Default time to wait for an acknowledgment after sending a command.
    DEFAULT_NAV_WIAT_TIME (float): Default time to wait for navigation data.

Example:
    ```python
    from serial import Serial
    from navigator.utility.ublox import Controller

    # Open a serial port
    serial_port = Serial("/dev/ttyACM0", 960, timeout=3)

    # Create a controller object
    controller = Controller(serial_port)
    ```
"""

import typing as tp
from logging import Logger
from queue import Queue
from socket import socket
from threading import Event, Lock, Thread
from time import sleep

import pyubx2 as ubx
from serial import Serial

__all__ = ["Controller"]

DEFAULT_COMMAND_WAIT_TIME = 0.001
DEFAULT_NAV_WIAT_TIME = 1.0


class Controller:
    """Controller class for the Ublox receiver.

    This class provides methods for controlling Ublox receivers through Serial or socket communication streams.

    Attributes:
        stream (Union[Serial, socket]): The communication stream with the Ublox receiver.
        logger (Optional[Logger]): The logger object to log I/O data.
        no_check (bool): If True, disables all default message rates.
        io_queue (Queue): Queue for storing received messages.
        ack_queue (Queue): Queue for storing acknowledgment messages.
        stop_event (Event): Event flag for stopping the I/O thread.
        lock (Lock): Thread lock for ensuring thread safety.
        io_thread (Thread): Thread for handling I/O operations.

    Example:
        ```python
        from serial import Serial
        from navigator.utility.ublox import Controller

        # Open a serial port
        serial_port = Serial("/dev/ttyACM0", 960, timeout=3)

        # Create a controller object
        controller = Controller(serial_port)
        ```
    """

    def __init__(
        self,
        stream: tp.Union[Serial, socket],
        logger: tp.Optional[Logger] = None,
        no_check: bool = False,
    ) -> None:
        """Initialize the Controller.

        Args:
            stream: The communication stream with the Ublox receiver.
            logger: The logger object to log I/O data. Defaults to None.
            no_check: If True, disables all default message rates. Defaults to False.

        Raises:
            TypeError: If the stream is not an instance of Serial or socket.
        """
        if not isinstance(stream, (Serial, socket)):
            raise TypeError("The stream should be an instance of Serial or socket.")

        self.stream: tp.Union[Serial, socket] = stream
        self.has_logger: bool = logger is not None
        self.logger: tp.Optional[Logger] = logger
        self.no_check: bool = no_check
        self.reader = ubx.UBXReader(
            datastream=self.stream, msgmode=ubx.GET, protfilter=ubx.UBX_PROTOCOL
        )

        # Queues for storing messages
        self.io_queue: Queue[ubx.UBXMessage] = Queue()
        self.ack_queue: Queue[ubx.UBXMessage] = Queue()

        # Event flag for stopping the I/O thread
        self.stop_event = Event()
        self.lock: Lock = Lock()

        # I/O thread for handling I/O operations
        self.io_thread: Thread = Thread(target=self.start_io_logic)
        self.io_thread.start()

        if not self.no_check:
            self.clear_output_rate()

    def clear_ack_queue(self) -> None:
        """Clear the acknowledgment queue."""
        # Acquire the lock to ensure thread safety
        with self.lock:
            while not self.ack_queue.empty():
                self.ack_queue.get()
        return

    def clear_io_queue(self) -> None:
        """Clear the I/O queue."""
        # Acquire the lock to ensure thread safety
        with self.lock:
            while not self.io_queue.empty():
                self.io_queue.get()
        return

    def start_io_logic(self) -> None:
        """Starts the I/O thread which reads the incoming data from the stream."""
        while not self.stop_event.is_set():
            if self.stream.in_waiting:
                try:
                    with self.lock:
                        _, parsed = self.reader.read()
                        if parsed is not None:
                            if parsed.identity.startswith("ACK"):
                                self.ack_queue.put(parsed)
                            else:
                                self.io_queue.put(parsed)
                            if self.has_logger:
                                self.logger.info(f"Received: {parsed.identity}")

                except Exception as e:
                    if self.has_logger:

                        self.logger.error(e)
                    continue

        return

    def send_control_command(self, command: ubx.UBXMessage) -> None:
        """Send a command to the Ublox receiver.

        Args:
            command: The command to be sent.
        """
        # Acquire the lock to ensure thread safety
        with self.lock:
            self.stream.write(command.serialize())

        # Log the command if a logger is provided
        if self.has_logger:
            self.logger.info(f"Sent: {command}")

        return

    def send_config_command(
        self, command: ubx.UBXMessage, wait_for_ack: bool = True
    ) -> tp.Optional[ubx.UBXMessage]:
        """Send a command to the Ublox receiver.

        Args:
            command: The command to be sent.
            wait_for_ack: If True, wait for an acknowledgment from the receiver. Defaults to True.

        Returns:
            The acknowledgment from the receiver if wait_for_ack is True, else None.
        """
        # Clear the acknowledgment queue first to avoid any stale data
        self.clear_ack_queue()
        # Write the command to the stream
        self.send_control_command(command)

        if wait_for_ack:
            # Wait for the acknowledgment
            while True:
                sleep(DEFAULT_COMMAND_WAIT_TIME)
                with self.lock:
                    if not self.ack_queue.empty():
                        return self.ack_queue.get()
        return None

    def send_poll_command(self, command: ubx.UBXMessage) -> None:
        """Send a command to the Ublox receiver.

        Args:
            command: The command to be sent.
        """
        # Send the command
        self.send_control_command(command)
        return

    def clear_output_rate(self) -> None:
        """Clear the output rate of all message types for MSGOUT."""
        cfg_msg = [msg for msg in ubx.UBX_CONFIG_DATABASE if "CFG_MSGOUT" in msg]

        for msg in cfg_msg:
            self.send_config_command(
                command=ubx.UBXMessage.config_set(
                    layers=ubx.SET_LAYER_RAM,
                    transaction=ubx.TXN_NONE,
                    cfgData=[(msg, 0)],
                ),
                wait_for_ack=True,
            )
        # Clear the queue
        self.clear_ack_queue()
        self.clear_io_queue()
        return

    def stop(self) -> None:
        """Stop the controller."""
        # Stop the I/O thread
        self.stop_event.set()
        self.io_thread.join()
        # user is responsible for closing the stream
        return

    def flush_messages(self) -> list[ubx.UBXMessage]:
        """Flush the available messages from the queue.

        Args:
            n: The number of messages to flush.

        Returns:
            The flushed messages.
        """
        # Acquire the lock to ensure thread safety
        with self.lock:
            messages = []
            while not self.io_queue.empty():
                messages.append(self.io_queue.get())

        return messages

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return f"Controller(stream={self.stream}, logger={self.logger}, no_check={self.no_check})"
