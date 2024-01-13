"""Idispatch interface for dispatching commands to the correct handler."""

from abc import ABC, abstractmethod

__all__ = ["Idispatch"]


class Idispatch(ABC):
    """Dispatch interface.

    Attributes:
        None

    Methods:
        dispatch: Dispatches a command to the correct handler.

    """

    @abstractmethod
    def dispatch(self, command: str) -> None:
        """Dispatches a command to the correct handler.

        Args:
            command: The command to dispatch.

        Returns:
            None

        Raises:
            NotImplementedError: This is an abstract class and should not be instantiated directly.

        """
        raise NotImplementedError
