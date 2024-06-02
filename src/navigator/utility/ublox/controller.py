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

import pyubx2 as ubx
from serial import Serial

__all__ = ["Controller"]

DEFAULT_COMMAND_WAIT_TIME = 0.1
DEFAULT_NAV_WIAT_TIME = 1.1


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

        self.__io_queue: Queue[ubx.UBXMessage] = Queue()
        self.__ack_queue: Queue[ubx.UBXMessage] = Queue()

        self._stop_event = Event()
        self._lock: Lock = Lock()

        self._io_thread: Thread = Thread(target=self._start_io_logic, daemon=True)
        self._io_thread.start()

        # Flush counter
        self.__flush_counter = 0

        if not self.no_check:
            self._clear_output_rate()

    def _start_io_logic(self) -> None:
        """Start the I/O thread."""
        reader: ubx.UBXReader = ubx.UBXReader(
            datastream=self.stream, msgmode=ubx.GET, protfilter=ubx.UBX_PROTOCOL
        )
        while not self._stop_event.is_set():
            if self.stream.in_waiting:
                try:
                    with self._lock:
                        _, parsed = reader.read()
                        if parsed is not None:
                            if parsed.identity.startswith("ACK"):
                                self.__ack_queue.put(parsed)
                            else:
                                self.__io_queue.put(parsed)
                                self.__flush_counter += 1

                            if self.has_logger:
                                self.logger.info(f"Received: {parsed}")

                except Exception as e:
                    if self.has_logger:
                        self.logger.error(e)
                    continue

    def _send_control_command(self, command: ubx.UBXMessage) -> None:
        """Send a command to the Ublox receiver.

        Args:
            command: The command to be sent.
        """
        if not self._stop_event.is_set():
            with self._lock:
                self.stream.write(command.serialize())

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
        # Clear the acknowledgment queue
        while not self.__ack_queue.empty():
            self.__ack_queue.get()

        # Write the command to the stream
        self._send_control_command(command)

        # Write the command to the stream
        if wait_for_ack:
            return self.__ack_queue.get(timeout=DEFAULT_COMMAND_WAIT_TIME)

        return None

    def send_poll_command(self, command: ubx.UBXMessage) -> None:
        """Send a command to the Ublox receiver.

        Args:
            command: The command to be sent.
        """
        self._send_control_command(command)
        return

    def _clear_output_rate(self) -> None:
        """Clear the output rate of all message types for MSGOUT."""
        cfg_msg = [msg for msg in ubx.UBX_CONFIG_DATABASE if "CFG_MSGOUT" in msg]

        for msg in cfg_msg:
            ack = self.send_config_command(
                command=ubx.UBXMessage.config_set(
                    layers=ubx.SET_LAYER_RAM,
                    transaction=ubx.TXN_NONE,
                    cfgData=[(msg, 0)],
                ),
                wait_for_ack=True,
            )

        if ack is None or ack.identity != "ACK-ACK":
            raise RuntimeError(
                """Unable to clear the output rate of all message types i.e the device should not show anything in message view in U-center.
                Manually set all message rates to 0 and use the no_check flag in the constructor."""
            )

        # Clear the queue
        while not self.__ack_queue.empty():
            self.__ack_queue.get()
        while not self.__io_queue.empty():
            self.__io_queue.get()
        return

    def stop(self) -> None:
        """Stop the controller."""
        # Stop the I/O thread
        self._stop_event.set()
        self._io_thread.join()

        # user is responsible for closing the stream
        return

    def flush_n_messages(self, n: int) -> list[ubx.UBXMessage]:
        """Flush n messages from the queue.

        Args:
            n: The number of messages to flush.

        Returns:
            The flushed messages.
        """
        messages = []
        with self._lock:
            if self.__flush_counter < n:
                raise ValueError(
                    "The number of messages to flush is greater than the number of messages in the queue."
                )

            for _ in range(n):
                messages.append(self.__io_queue.get())

            self.__flush_counter -= n

        return messages

    def __len__(self) -> int:
        """Return the number of messages in the queue."""
        with self._lock:
            return self.__flush_counter

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return f"Controller(stream={self.stream}, logger={self.logger}, no_check={self.no_check})"
